import matplotlib.pyplot as plt
%matplotlib inline

def plot_train_stat_per_epoch(history):
    acc = history['binary_accuracy']
    val_acc = history['val_binary_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)

    f, axarr = plt.subplots(1, 2, figsize=(18, 6))
    axarr[0].plot(epochs, acc, 'bo', label='Training acc')
    axarr[0].plot(epochs, val_acc, 'b', label='Validation acc')
    axarr[0].set_title('Training and validation accuracy')
    axarr[0].legend()

    axarr[1].plot(epochs, loss, 'bo', label='Training loss')
    axarr[1].plot(epochs, val_loss, 'b', label='Validation loss')
    axarr[1].set_title('Training and validation loss')
    axarr[1].legend()
    
def one_hot_encode_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results