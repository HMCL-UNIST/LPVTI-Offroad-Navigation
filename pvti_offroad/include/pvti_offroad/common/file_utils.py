import os
import pickle
import pathlib

sim_dir = os.path.expanduser('~') + '/offroad_sim_data/'
train_dir = os.path.join(sim_dir, 'trainingData/')
preprocessed_dir = os.path.join(sim_dir, 'trainingData/preprocessed/')
eval_dir = os.path.join(sim_dir, 'evaluationData/')
model_dir = os.path.join(sim_dir, 'models/')
snapshot_dir = os.path.join(sim_dir, 'snapshot/')

train_log_dir = os.path.join(sim_dir, 'train_log/')


def dir_exists(path=''):
    dest_path = pathlib.Path(path).expanduser()
    return dest_path.exists()


def create_dir(path='', verbose=False):
    dest_path = pathlib.Path(path).expanduser()
    if not dest_path.exists():
        dest_path.mkdir(parents=True)
        return dest_path
    else:
        if verbose:
            print('- The source directory %s does not exist, did not create' % str(path))
        return None


def pickle_write(data, path):
    dbfile = open(path, 'wb')
    pickle.dump(data, dbfile)
    dbfile.close()


def pickle_read(path):
    dbfile = open(path, 'rb')
    data = pickle.load(dbfile)
    dbfile.close()
    return data

if not dir_exists(train_log_dir):
    create_dir(train_log_dir)

if not dir_exists(model_dir):
    create_dir(model_dir)
    
if not dir_exists(snapshot_dir):
    create_dir(snapshot_dir)