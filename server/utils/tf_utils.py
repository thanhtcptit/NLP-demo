import os
import tensorflow as tf

from tensorflow.python.training import training, saver
from tensorflow.python.platform import gfile


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError(
            'No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError(
            'There should not be more than one meta file in (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_model(model, input_map=None, session=None):
    """ Check if the model is a model directory
    (containing a metagraph and a checkpoint file)
    or if it is a protobuf file with a frozen graph """
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(
            os.path.join(model_exp, meta_file), input_map=input_map)
        if session:
            saver.restore(session,
                          os.path.join(model_exp, ckpt_file))
        else:
            saver.restore(tf.get_default_session(),
                          os.path.join(model_exp, ckpt_file))


def _get_checkpoint_filename(filepattern):
    """Returns checkpoint filename given directory or specific filepattern."""
    if gfile.IsDirectory(filepattern):
        return saver.latest_checkpoint(filepattern)
    return filepattern


def load_checkpoint(filepattern):
    """Returns CheckpointReader for latest checkpoint.
    Args:
    filepattern: Directory with checkpoints file or path to checkpoint.
    Returns:
    `CheckpointReader` object.
    Raises:
    ValueError: if checkpoint_dir doesn't have 'checkpoint'
    file or checkpoints.
    """
    filename = _get_checkpoint_filename(filepattern)
    if filename is None:
        raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                         "given directory %s" % filepattern)
    return training.NewCheckpointReader(filename)


def list_variables(checkpoint_dir):
    """Returns list of all variables in the latest checkpoint.
    Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.
    Returns:
    List of tuples `(name, shape)`.
    """
    reader = load_checkpoint(checkpoint_dir)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    result = []
    for name in names:
        result.append((name, variable_map[name]))
    return result


def get_all_scopes(checkpoint_dir):
    var_list = list_variables(checkpoint_dir)
    scopes = set()
    for name, dim in var_list:
        scope = name.split('/')[0]
        scopes.add(scope)
    return scope


if __name__ == '__main__':
    list_vars = list_variables('train_logs/multi_cased_fim_finetune/'
                               'predict_model')
    for v in list_vars:
        print(v)
