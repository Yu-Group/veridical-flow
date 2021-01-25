'''Class that stores the entire pipeline of steps in a data-science workflow
'''
import joblib
from sklearn.pipeline import Pipeline

class PCSPipeline(Pipeline):
    def __init__(self):
        self.preprocess = False

    def run(self):
        '''Runs the pipeline

        '''
        pass
