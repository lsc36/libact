"""JSRE

An interface for jSRE Classifier model.
"""
import sys
import os
import subprocess
import re
import numpy as np
import logging
import datetime
import time
LOGGER = logging.getLogger(__name__)

from libact.base.interfaces import ProbabilisticModel


class JSRE(ProbabilisticModel):

    """Wrapper for jSRE classifier utilizing a SVM with a shallow lingusitics kernel

    References
    ----------
    https://hlt-nlp.fbk.eu/technologies/jsre

    """

    def __init__(self, model_path, jsre_path, tmp_dir='/tmp'):
        """model_path => where to save the model to"""
        self.tmp_dir = tmp_dir
        self.model = model_path
        self.classpath = './bin:./lib/*'.format(jsre_path, jsre_path)
        self.jsre_path = jsre_path
        self.max_memory = '-mx1024M'
        self.predict_template = 'java -cp {cp} {memory} org.itc.irst.tcc.sre.Predict {to_predict} {model} {output}'
        self.train_template = 'java -cp {cp} {memory} org.itc.irst.tcc.sre.Train -m 512 -k SL -n 3 -w 2 {to_train} {model_output}'

    def __generate_timestamp(self):
        return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

    def __get_tmp_filename(self, string):
        ts = self.__generate_timestamp()
        return os.path.join(self.tmp_dir, '{}-{}.jsre'.format(string, ts))

    def __run_command(self, cmd, output_file=None):
        try:
            pc = subprocess.run(cmd.split(' '), check=True, stdout=subprocess.PIPE,
                                cwd=self.jsre_path)
            if output_file is None:
                return None, pc.stdout.decode('utf-8')
            with open(output_file, 'r') as outputf:
                predictions = outputf.read().strip().split('\n')
            return predictions, pc.stdout
        except subprocess.CalledProcessError as e:
            LOGGER.error('Could not run "{}": {}'.format(cmd, e))
            sys.exit(-1)
        # except FileNotFoundError:
        #     LOGGER.error('Could not find output file "{}".'.format(output_file))
        #     sys.exit(-1)

    def __raw_predict(self, instances):
        ts = self.__generate_timestamp()
        tmp_example_file = self.__get_tmp_filename('test')
        tmp_output_file = self.__get_tmp_filename('output')

        with open(tmp_example_file, 'w') as testf:
            for instance in instances:
                # use 1 as default label b/c -1 will crash jSRE even tough thats the proposed label for unknown
                testf.write('1\t{}\n'.format(instance[0]))  # all features are encoded as one for jSRE

        cmd = self.predict_template.format(
            cp=self.classpath, memory=self.max_memory, model=self.model,
            to_predict=tmp_example_file, output=tmp_output_file)

        raw_prediction, _ = self.__run_command(cmd, tmp_output_file)
        splitted_lines = (line.split('\t') for line in raw_prediction[1:])
        predictions, probas = zip(*[(int(float(row[0])), [float(n) for n in row[2:]]) for row in splitted_lines])
        return predictions, probas

    def train(self, dataset, *args, **kwargs):
        ts = self.__generate_timestamp()
        tmp_training_file = self.__get_tmp_filename('train')
        lines = ['{}\t{}'.format(lbl, feat) for feat, lbl in zip(*dataset.format_jsre())]
        with open(tmp_training_file, 'w') as trainingf:
            trainingf.write('\n'.join(lines))

        cmd = self.train_template.format(
            cp=self.classpath, memory=self.max_memory, model_output=self.model,
            to_train=tmp_training_file)

        self.__run_command(cmd)
        try:
            os.remove(tmp_training_file)
        except OSError:
            pass

    def predict(self, instances, *args, **kwargs):
        predictions, _ = self.__raw_predict(instances)
        return np.array(predictions)

    def predict_proba(self, instances, *args, **kwargs):
        _, probas = self.__raw_predict(instances)
        # this is dependent on the data and can be the other way round as well!
        return np.array(probas)

    def score(self, dataset, *args, **kwargs):
        tmp_test_file = self.__get_tmp_filename('test')
        tmp_output_file = self.__get_tmp_filename('output')
        lines = ['{}\t{}'.format(lbl, feat) for feat, lbl in zip(*dataset.format_jsre())]
        with open(tmp_test_file, 'w') as trainingf:
            trainingf.write('\n'.join(lines))

        cmd = self.predict_template.format(
            cp=self.classpath, memory=self.max_memory, model=self.model,
            to_predict=tmp_test_file, output=tmp_output_file)
        _, stdout = self.__run_command(cmd)

        match = re.search(r'Accuracy = (\d+\.\d+)%', stdout)
        if match is None or len(match.groups()) < 1:
            LOGGER.error('Could not extract accuracy from output of "{}"'.format(cmd))
            sys.exit(-1)

        accuracy = float(match.groups()[0])/100
        try:
            os.remove(tmp_test_file)
            os.remove(tmp_output_file)
        except OSError:
            pass

        return accuracy