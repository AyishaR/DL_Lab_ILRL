import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch
import pprint
import sys
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.append(".")

import utils
from imitation_learning.agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation


def split_proportionally(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=None, stratify=y
    )

    return X_train, y_train, X_valid, y_valid


def read_data(filename, datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, filename)

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    try:
        X = np.array(data["state"]).astype("float32")
        y = np.array(data["action"]).astype("float32")
    except ValueError:
        X, y = [], []
        for i in range(len(data["state"])):
            X.extend(data["state"][i])
            y.extend(data["action"][i])
        X = np.vstack(data["state"]).astype("float32")
        y = np.vstack(data["action"]).astype("float32")
    y = np.apply_along_axis(utils.action_to_id, 1, y)

    return X, y


def augment(X, y, custom_sizes):
    subsets = {}
    for label in np.unique(y):
        subsets[label] = X[y == label]

    X_augmented = []
    y_augmented = []
    for label, data in subsets.items():
        size = custom_sizes.get(label, len(data))
        if size > len(data):
            augmented_data = np.concatenate([data] * (size // len(data)))
            remainder = size % len(data)
            if remainder > 0:
                augmented_data = np.concatenate([augmented_data, data[:remainder]])
        else:
            indices = np.random.choice(len(data), size=size, replace=False)
            augmented_data = data[indices]

        X_augmented.append(augmented_data)
        y_augmented.append(np.array([label] * len(augmented_data)))

    X_augmented = np.concatenate(X_augmented, axis=0)
    y_augmented = np.concatenate(y_augmented, axis=0)

    X_augmented, y_augmented = shuffle(X_augmented, y_augmented)

    return X_augmented, y_augmented


def stack_and_reshape(array, N):
    num_elements = array.shape[0]
    B = num_elements - N + 1
    stacked_views = [array[i : i + N, :, :] for i in range(B)]
    reshaped_array = np.stack(stacked_views, axis=0)
    return reshaped_array


def preprocessing(X, y, history_length=0):
    X = utils.rgb2gray(X)
    X[:, 85:, :15] = 0.0
    if history_length >= 0:
        X = stack_and_reshape(X, history_length + 1)
        y = y[history_length:]

    return X, y


def get_train_test(X, y, custom_sizes={}):
    X_train, y_train, X_valid, y_valid = split_proportionally(X, y)

    class_distribution = {cls: np.sum(y == cls) for cls in range(5)}
    print("Before augment")
    pprint.pprint(class_distribution)

    X_train, y_train = augment(X_train, y_train, custom_sizes)

    print("After augment")
    cd_train = {cls: np.sum(y_train == cls) for cls in range(5)}
    cd_valid = {cls: np.sum(y_valid == cls) for cls in range(5)}
    pprint.pprint(cd_train)
    pprint.pprint(cd_valid)

    return X_train, y_train, X_valid, y_valid


def train_model(
    X,
    y,
    device,
    n_minibatches,
    batch_size,
    lr,
    epochs=2,
    history_length=0,
    model_dir="./models_im",
    tensorboard_dir="./tensorboard",
    agent_path=None,
    model_name="agent",
):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    agent = BCAgent(history_length=history_length, lr=lr, n_classes=5)
    if agent_path:
        agent.load(agent_path)

    tensorboard_eval = Evaluation(
        tensorboard_dir,
        f"Imitation Learning ",
        stats=["train_accuracy", "train_loss", "valid_accuracy", "valid_loss"],
    )

    def sample_minibatch():
        indices = np.random.permutation(len_X_train)
        minibatch_indices = indices[:batch_size]

        X_batch = X_train[minibatch_indices]
        y_batch = y_train[minibatch_indices]

        return X_batch, y_batch

    X, y = preprocessing(X, y, history_length=history_length)

    for epoch in range(epochs):

        X_train, y_train, X_valid, y_valid = get_train_test(
            X, y, custom_sizes=custom_sizes
        )

        len_X_train = X_train.shape[0]
        y_valid_tensor = torch.tensor(y_valid)

        for i in range(n_minibatches):
            X_batch, y_batch = sample_minibatch()
            train_loss = agent.update(X_batch, y_batch).item()

            if i % 10 == 0 or i == n_minibatches - 1:
                _, y_pred = agent.predict(X_batch)
                y_pred = y_pred.numpy()
                train_accuracy = (y_pred == y_batch).sum() / len(y_batch)

                y_pred_raw, y_pred = agent.predict(X_valid)
                y_pred = y_pred.numpy()
                val_loss = agent.loss_fn(y_pred_raw, y_valid_tensor).item()

                valid_accuracy = (y_pred == y_valid).sum() / len(y_valid)

                eval_dict = {
                    "train_accuracy": train_accuracy,
                    "train_loss": train_loss,
                    "valid_accuracy": valid_accuracy,
                    "valid_loss": val_loss,
                }

                tensorboard_eval.write_episode_data(i, eval_dict)

                agent.save(os.path.join(model_dir, model_name))
                print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    device = torch.device("cpu")
    print(device)

    custom_sizes = {
        0: 3000,
        1: 3000,
        2: 3000,
        3: 3000,
        4: 200,
    }

    # read data
    X, y = read_data("data5.pkl.gzip", "./data")

    # agent_path = "./models_im/agent_data3_h5_v17.pt"
    agent_path = None

    for history_length in [0, 1, 3, 5]:

        # train model (you can change the parameters!)
        train_model(
            X,
            y,
            device,
            n_minibatches=300,
            batch_size=64,
            lr=1e-4,
            history_length=history_length,
            agent_path=agent_path,
            epochs=4,
            model_name=f"agent_data5_h{history_length}_v18.pt",
            model_dir="./models_final_im",
        )
