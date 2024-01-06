import csv

def load_data(data_file):
    data = []
    with open(data_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)

        for row in csv_reader:
            data.append(row)
    data = np.array(data, dtype=np.float64)
    return data

def show_image(X, y, index):
    image = np.array(X[index], dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    print("Label: " + str(y[index]))
