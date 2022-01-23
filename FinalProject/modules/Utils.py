import pickle
import requests
import tarfile
import os
import numpy as np
import tensorflow as tf
from IPython.display import display, HTML

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
'''
1. Pickle is used for the process of converting a Python object into a byte stream to store
it in a file/database, maintain program state across sessions, or transport data over the network.
2. IPython.display is a public API for display tools in IPython.
3. BASE_DIR is used to return the directory name of pathname path. This is the first element 
of the pair returned by passing path to the function split(). os.path.realpath will first resolve
any symbolic links in the path, and then return the absolute path.
'''

def download_cifar10(download_100 = False):
    '''
    Download cifar-10 tarzip file and unzip for using,
    Args:
        download_100 = True ==> download cifar-100 data set.
        download_100 = False ==> download cifar-10 data set.
    then using os.path.join() method to join various path components.
    '''
    if download_100:
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    else:
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = url.split("/")[-1]
    fpath = os.path.join(BASE_DIR, filename)

    # In case of duplication:
    if os.path.exists(fpath):
        print('file already downloaded..')
        return

    # download cifar-10 file:
    # 'wb' represents for write and binary
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    # unzip tarzip file:
    # "r:gz" represents open for reading with gzip compression
    tar = tarfile.open(filename, "r:gz")
    # Extract all members from the archive to the current working directory or directory path.
    tar.extractall()
    tar.close()


def load_cifar10(num_batches=5, get_test_data=True, channels_last=True):
    '''
    Load the cifar-10 data.
    Args:
        num_batches: int, the number of batches of data to return.
        get_test_data: bool, whether to return test data.
        Channel_last: bool, set to control the structure for 'NHWC' OR 'NCHW'.
    Returns:
        If get_test_data False ==> (images, labels).
        Otherwise ==> (images, labels, test_images, test_labels).
        images are numpy arrays of shape: (num_images, num_channels, width, height)'NCWH'.
        labels are 1D numpy arrays contains labels correlated to train data.
    '''
    assert num_batches <= 5
    # download if file itself not exists:
    download_cifar10()

    # load batches in order with directory of :
    dirpath = os.path.join(BASE_DIR, 'cifar-10-batches-py')
    images = None # 'None' means
    for i in range(1, num_batches + 1):
        print('getting batch {0}'.format(i))
        filename = 'data_batch_{0}'.format(i)
        fpath = os.path.join(dirpath, filename)

        with open(fpath, 'rb') as f:
            # "r" - Read - Default value. Opens a file for reading, error if the file does not exist
            # "a" - Append - Opens a file for appending, creates the file if it does not exist
            # "w" - Write - Opens a file for writing, creates the file if it does not exist
            # "x" - Create - Creates the specified file, returns an error if the file exist
            # The "rb" mode opens the file in binary format for reading
            content = pickle.load(f, encoding='bytes')
        if images is None:
            images = content[b'data']
            labels = content[b'labels']
        else:
            # Stack arrays in sequence vertically (row wise).
            images = np.vstack([images, content[b'data']])
            # The extend() method adds all the elements of an iterable to the end of the list.
            labels.extend(content[b'labels'])
    # convert to labels:
    labels = np.asarray(labels)
    # convert to RGB format with 3 channels:
    images = images.reshape(-1, 3, 32, 32)
    # normalize data by dividing by 255:
    images = images / 255.
    if channels_last:
        # Move axes of an array to new positions. ==> 'NHWC'
        images = np.moveaxis(images, 1, -1)

    if not get_test_data: # No test data
        return images, labels

    filename = 'test_batch'
    fpath = os.path.join(dirpath, filename)
    with open(fpath, 'rb') as f:
        content = pickle.load(f, encoding='bytes')
    test_images = content[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.asarray(content[b'labels']) # Convert the input to an array

    # Normalize data by dividing by 255:
    test_images = test_images / 255.
    # Make channels last: ==> NHWC
    if channels_last:
        test_images = np.moveaxis(test_images, 1, -1)

    return images, labels, test_images, test_labels


def load_cifar100(get_test_data=True, channels_last=True):
    """
    Load the cifar 100 data (not in batches).
    Args:
        get_test_data: bool, whether to return test data.
        Channel_last: bool, set to control the structure for 'NHWC' OR 'NCHW'.
    Returns:
        If get_test_data False ==> (images, labels).
        Otherwise ==> (images, labels, test_images, test_labels).
        images are numpy arrays of shape: (num_images, num_channels, width, height)'NCWH'.
        labels are 1D numpy arrays contains labels correlated to train data.
    """
    # Save memory for less function definition
    download_cifar10(download_100=True)

    # load batches in order:
    dirpath = os.path.join(BASE_DIR, 'cifar-100-python')
    images = None # No test data return
    filename = 'train'
    fpath = os.path.join(dirpath, filename)
    with open(fpath, 'rb') as f:
        content = pickle.load(f, encoding='bytes')
    if images is None:
        images = content[b'data']
        labels = content[b'fine_labels']
    # convert to labels:
    labels = np.asarray(labels)
    # convert to RGB format with 3 channels:
    images = images.reshape(-1, 3, 32, 32)

    # normalize data by dividing by 255:
    images = images / 255.
    if channels_last:
        images = np.moveaxis(images, 1, -1)

    if not get_test_data:
        return images, labels

    filename = 'test'
    fpath = os.path.join(dirpath, filename)
    with open(fpath, 'rb') as f:
        content = pickle.load(f, encoding='bytes')
    test_images = content[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.asarray(content[b'fine_labels'])

    # normalize data by dividing by 255:
    test_images = test_images / 255.
    # Make channels last: ==> NHWC
    if channels_last:
        test_images = np.moveaxis(test_images, 1, -1)

    return images, labels, test_images, test_labels


def strip_consts(graph_def, max_const_size=32):
    '''
    Strip large constant values from graph_def.
    '''
    strip_def = tf.GraphDef() # A protobuf containing the graph of operations.
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0) # Merge the fields from the given message into this message.
        if n.op == 'Const':
            tensor = n.attr['value'].tensor # Access tensor_content values in TensorProto
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>" % size
    return strip_def


def show_graph(graph_def, max_const_size=32):
    '''
    Visualize TensorFlow graph.
    '''
    # The hasattr() method returns true if an object has the given named attribute and false if it does not.
    if hasattr(graph_def, 'as_graph_def'):
        # as_graph_def() returns a serialized GraphDef representation of this graph.
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size = max_const_size)

    code = """
        <script src="//cdnjs.cloudflare.com/ajax/libs/polymer/0.3.3/platform.js"></script>
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1000px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe)) # An inline frame is used to embed another document within the current HTML document.