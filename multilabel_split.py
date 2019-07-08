"""

Module for splitting data into train and test sets (e.g. a simple 70%-30% split) for multilabel datasets.

This algorithm ensures that the distribution of labels is kept for all the labels simultaneously. This is a well-known problem in
multi-label classification, but to the best of our knowledge there is no implementation available that can scale to GB scale
(millions of instances, hundreds of features, more than a hundred classes).

The algorithm is inspired on the following proposal (with some assumptions simplified in order to make the proposal scalable):

Sechidis, K.,Tsoumakas, G., Vlahavas, I.: On the stratification of multi-label data.
Machine Learning and Knowledge Discovery in Databases pp. 145-158 (2011)
available: http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf

It uses a iterative Greedy strategy to do the partitioning, focusing at every step on the class with less unnasigned instances in the data set.
These are split and assigned randomly to the train/test sets. Then, a new iterations begins.

The algorithm keeps looping over the data set, tagging instances as TRAIN/TEST until all classes have been scanned. Then, a final pass filter
the dataset for extracting the final partitions.

There are some improvements that could be implemented to increase the accuracy of the split. The most important would be to update the
splitting the features using a dynamic ratio (updated accordingly based on the current distribution of the instances already assigned).
However, the current results are good enough to use the algorithm as is.

There is also a fair amount of stats computation at the end that could be removed for speeding up things.

Example:

spark-submit --master=yarn --conf spark.driver.maxResultSize=3g --conf spark.serializer=org.apache.spark.serializer.KryoSerializer
--conf spark.kryoserializer.buffer.max=2047 --conf spark.akka.frameSize=2047 --conf spark.rdd.compress=true partition_data.py
--output_train <some S3 address>
--output_test <some S3 address> 
--mappings_path <some S3 address>
--input_path <some S3 address>

On 50 r3.2xlarge, and for about 40 GBs of data, it should take 20-30 minutes

"""

import pickle
import logging
from argparse import ArgumentParser
from ast import literal_eval
from collections import defaultdict
from operator import add

from pyspark.mllib.random import RandomRDDs
from pyspark.sql import SparkSession

# configure logging
lgg = logging.getLogger(__name__)

session = SparkSession.builder.getOrCreate()
sc = session.sparkContext
sc.setLogLevel("FATAL")

# define tags for labelling instances in the data set
UNASSIGNED = -1
TRAIN = 0
TEST = 1


TRAIN_RATIO = 0.7   # ratio of instances in the training set
MIN_INSTANCES = 20  # minimum number of instances to consider a class
SEED = 12345


def get_parsing_function(broadcasted_mappings):
    """ Define a parsing function for the data

    Args:
        broadcasted_mappings (Broadcasted variable):

    Returns:
        (function) to parse rows from the initial data set
    """
    # evaluate the row strings, map labels to indexes and label all instances as unassigned
    # return a function that does that

    def parse_row(spark_row):
        """ Parse a row of data

            Eval the string into a tuple of (features, labels), map each label to its corresponding index
            and tag the instance as UNASSIGNED

        Args:
            spark_row (Row): Row of data

        Returns:
            (list, list, TAG-int) respresenting an instance (features, labels, state)
        """

        data = literal_eval(spark_row.value)
        features = data[0]
        mappings = broadcasted_mappings.value
        labels = [mappings[l] for l in data[1] if l in mappings]

        output = (features, labels, UNASSIGNED)

        return output

    return parse_row


def count_classes(input_rows):
    """ Count how many instances of each class are in this partition

    Args:
        input_rows (Spark partition): iterable with data set instances ((features,labels,status), probability)

    Returns:
        (iterator) for class counts, formated as (class,num_instances)
    """

    counts = defaultdict(int)

    for row in input_rows:
        data = row[0]
        if data[2] == UNASSIGNED:
            classes = data[1]
            for _class in classes:
                counts[_class] += 1

    output = [(_class, counts[_class]) for _class in counts]

    return iter(output)


def count_final_classes(input_rows):
    """ Count how many instances of each class are in this partition

    Args:
        input_rows (Spark partition): iterable with data set instances (features,labels,status)

    Returns:
        (iterator) for class counts, formated as (class,num_instances)
    """

    counts = defaultdict(int)

    for row in input_rows:
        classes = row[1]
        for _class in classes:
            counts[_class] += 1

    output = [(_class, counts[_class]) for _class in counts]

    return iter(output)


def compute_stats(dataset_rdd, final_stats=False):
    """ Count how many instances of each class are in the data set

        Uses MapPartitions to speed up the counting

    Args:
        dataset_rdd (RDD): RDD with the data set
        final_stats (bool): Whether the data is in the final output format or not

    Returns:
        (dict(int, int)) with counts of instances for each class
    """

    count_function = count_final_classes if final_stats else count_classes

    # accumulate counts per class
    counts = dataset_rdd.mapPartitions(count_function)
    stats = {_class: count for _class, count in counts.reduceByKey(add).collect()}

    return stats


def assign_instance(row, target_class):
    """ Label an instance as TRAIN or TEST if the target class is one of its labels

    Args:
        row (tuple): Instance in the data set
        target_class (int): Class that is being partitioned at the moment

    Returns:
        (tuple): The row with its label updated if one if its labels was the target class
    """

    instance, prob = row

    if target_class in instance[1] and instance[2] == UNASSIGNED:
        is_train = prob < TRAIN_RATIO
        tag = TRAIN if is_train else TEST

        row = (instance[0], instance[1], tag), prob

    return row


def main(settings):
    """ Main method of the algorithm

    Args:
        settings (dict(str, object)): Command line parameters

    """

    # 1.1 Read label mappings and define the parsing function
    with open(settings['mappings_path'], "rb") as infile:
        label_id_to_idx_mapping = pickle.load(infile)

    last_class = max(_class for _class in label_id_to_idx_mapping)

    mappings_b = session.sparkContext.broadcast(label_id_to_idx_mapping)
    parsing_function = get_parsing_function(mappings_b)

    # 1.2 Read & parse input data
    data = session.read.text(settings['input_path'])

    data = data.rdd.map(parsing_function)
    all_count = data.count()

    lgg.info("Number of input instances: {}".format(all_count))

    # 2.1 generate exactly one random number (probability) for each instance
    # -- as an exercise, try to answer why do this now, and why we only need one :evil:
    data_partitions = data.getNumPartitions()

    random_numbers = RandomRDDs.uniformRDD(session.sparkContext, all_count, seed=SEED, numPartitions=data_partitions)

    # 2.2 Join the probabilities and the data set
    # use the zipWithIndex + join trick, because zip simply doesn't work ;(
    data = data.zipWithIndex().map(lambda (row, _id): (_id, row))
    random_numbers = random_numbers.zipWithIndex().map(lambda (row, _id): (_id, row))

    data_with_probabilities = data.join(random_numbers).map(lambda (index, data): data)

    lgg.info("Probabilities computed")

    # 3.1 Collect initial stats ...
    stats = compute_stats(data_with_probabilities)

    # 3.2 ... and blacklist classes with very few examples
    blacklist = {_class for _class, count in stats.iteritems() if count < MIN_INSTANCES}
    stats = {_class: count for _class, count in stats.iteritems() if _class not in blacklist}

    lgg.info("Blacklisting {} classes".format(len(blacklist)))

    # 4.1 Main loop - iterate until there are no classes with examples unassigned
    while stats:
        # 4.1.1 Choose the class with the minimum number of instances
        min_class, count = next(iter(sorted(stats.items(), key=lambda x: x[1])))

        lgg.info("Distributing class {} with {} instances".format(min_class, count))

        # 4.1.2 Assign instances of that class
        data_with_probabilities = data_with_probabilities.map(lambda x: assign_instance(x, min_class))

        # 4.1.3 Check new stats
        stats = compute_stats(data_with_probabilities)
        stats = {_class: count for _class, count in stats.iteritems() if _class not in blacklist}
        data_with_probabilities.cache()

    # 5. Separate the 3 final sets of instances
    train_set = data_with_probabilities.filter(lambda ((feat, _class, tag), prob): tag == TRAIN).map(
        lambda ((feat, _class, tag), prob): (feat, _class))
    test_set = data_with_probabilities.filter(lambda ((feat, _class, tag), prob): tag == TEST).map(
        lambda ((feat, _class, tag), prob): (feat, _class))
    # This one is not really necessary - used only for counting how many instances will be discarded
    free_set = data_with_probabilities.filter(lambda ((feat, _class, tag), prob): tag == UNASSIGNED).map(
        lambda ((feat, _class, tag), prob): (feat, _class))

    # 6. Show stats on the final split
    tr_count = train_set.count()
    ts_count = test_set.count()
    fs_count = free_set.count()

    lgg.info("Training: {}, Test: {}, Free: {}".format(tr_count, ts_count, fs_count))

    stats_train = compute_stats(train_set, final_stats=True)
    stats_test = compute_stats(test_set, final_stats=True)

    lgg.info("Final distribution")
    for f in range(last_class + 1):

        tr = stats_train.get(f, 0)
        ts = stats_test.get(f, 0)

        if tr < 1:
            continue

        ratio = float(tr) / float(tr + ts)

        lgg.info("Class {}, Train: {}, Test: {}, Ratio: {}".format(f, tr, ts, ratio))

    # 7. Write training and test sets
    train_set.saveAsTextFile(settings['output_train'])
    test_set.saveAsTextFile(settings['output_test'])


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--mappings_path', required=True)
    parser.add_argument('--output_train', required=True)
    parser.add_argument('--output_test', required=True)

    args = vars(parser.parse_args())

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)