from sklearn.externals import joblib


# save a model to a file
def save(model, filename):
    joblib.dump(model, filename + '.pkl')


# load a model from a file
def load(saved_model_filename):
    model = joblib.load(saved_model_filename + '.pkl')
    return model
