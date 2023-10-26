import cv2
import glob
import numpy as np
import tensorflow as tf
import json
from copy import deepcopy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Ova funkcija služi za definisanje training dataseta.


def make_training_set():
    training_set = tf.keras.utils.image_dataset_from_directory(
        'C:\\Users\\Oli\\Documents\\Faks\\2\\seminarski b\\aicook.v1-original-images.yolov8\\train\\bestAugmented128',
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=5555555,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    return training_set


# Ova funkcija služi za definisanje validation dataseta.


def make_validation_set():
    validation_set = tf.keras.utils.image_dataset_from_directory(
        'C:\\Users\\Oli\\Documents\\Faks\\2\\seminarski b\\aicook.v1-original-images.yolov8\\valid\\bestAugmented128',
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=5555555,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    return validation_set


# Ova funkcija služi za definisanje samog modela.


def make_model(training_set, validation_set):
    with tf.device('/GPU:0'):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.7))
        model.add(Dense(30, activation='softmax'))

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        history = model.fit(
            x=training_set,
            validation_data=validation_set,
            epochs=25
        )

        model.save('C:\\Users\\Oli\\PycharmProjects\\Fridger\\Models\\justatest.h5')

        with open('training_history.json', 'w') as f:
            json.dump(history.history, f)

        return model

# Ova funkcija služi za testiranje određenog test priemra na modelu.

def test_case(br, model):
    # Ovaj deo funkcije služi za unos i predprocesovanje test primera.
    global dimx, dimy
    files = glob.glob(f"C:\\Users\\Oli\\Documents\\Faks\\2\\seminarski b\\PineTools.com_2023-05-07_11h54m19s\\PineTools.com_files\\*.jpg")

    dict = {0: 'apple', 1: 'banana', 2: 'beef', 3: 'blueberries', 4: 'bread', 5: 'butter', 6: 'carrot', 7: 'cheese',
            8: 'chicken', 9: 'chicken_breast',
            10: 'chocolate', 11: 'corn', 12: 'eggs', 13: 'flour', 14: 'goat_cheese', 15: 'green_beans',
            16: 'ground_beef', 17: 'ham', 18: 'heavy_cream', 19: 'lime',
            20: 'milk', 21: 'mushrooms', 22: 'onion', 23: 'potato', 24: 'shrimp', 25: 'spinach', 26: 'strawberries',
            27: 'sugar', 28: 'sweet_potato', 29: 'tomato'}

    # Skaliramo sliku na 30% njene originalne veličine radi efikasnosti.

    scale_percent = 30 / 100
    dimx = int(cv2.imread(files[br]).shape[0] * scale_percent)
    dimy = int(cv2.imread(files[br]).shape[1] * scale_percent)

    image = cv2.imread(files[br])
    orig = image
    backup = deepcopy(orig)

    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # value = 180 se pokazala kao najbolja vrednost.
    # U ovom delu se povećava vrednost brightness-a pixela.

    vvalue = 180
    vlim = 255 - vvalue
    v[v > vlim] = 255
    v[v <= vlim] += vvalue

    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # border = 205 se pokazala kao najbolja vrednost
    # U ovom delu pixeli dobijaju vrednost 255 ili 0 u zavisnosti od vrednosti promenljive "border".

    border = 205
    ret, thresh = cv2.threshold(gray, border, 255, cv2.THRESH_BINARY_INV)
    fridge = np.asarray(thresh)
    fridge = fridge.astype(int)

    cv2.imshow('Black and White', thresh)
    cv2.waitKey(0)
    cv2.imshow('Original', cv2.resize(orig, (400, 650)))
    cv2.waitKey(0)

    """
    U ovom delu funkcije se obavlja pronalaženje i predviđanje objekata na slici.
    "lower_bound" predstavlja minimalnu vrednost sume submatrice da bi se uzimala u obzir za predviđanje.
    "bg_factor" predstavlja "kaznu" algoritmu kada uzima pixel koji nije objekat.
    Ova stavka je neophodna, jer bi inače algoritam samo uzeo celu sliku.
    "l" i "step" predstavljaju offset i korak. Kada se nađe potencijlni objekat na slici,
    algoritam traži na tom mestu i u okruženju koje zavisi od ovih parametara.
    """

    lower_bound = 3000
    bg_factor = 75
    l = 10
    step = 4

    while True:
        fridge = fridge - bg_factor
        coordinates = find_max_sum(fridge)
        top_left = coordinates[:2]
        bottom_right = coordinates[2:4]
        sum = coordinates[4]

        if sum >= lower_bound:
            best_answer = 0

            # U ovom delu se ne gleda tačno "okruženje" predviđanja, već ubliženu i udaljenu verziju

            for d in range(-l, l, step):
                tlx = max(top_left[0] - d, 0)
                brx = min(bottom_right[0] + d, dimx - 1)
                tly = max(top_left[1] - d, 0)
                bry = min(bottom_right[1] + d, dimy - 1)

                if tlx < brx and tly < bry:
                    og_tlx = int(tlx / scale_percent)
                    og_tly = int(tly / scale_percent)
                    og_brx = int(brx / scale_percent)
                    og_bry = int(bry / scale_percent)

                    sample = orig[og_tlx:og_brx, og_tly:og_bry]
                    sample = cv2.resize(sample, (128, 128), interpolation=cv2.INTER_LINEAR)
                    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                    sample = np.array([sample])

                    pr = model.predict(sample, verbose=0)

                    if pr.max() > best_answer:
                        x1 = og_tlx
                        y1 = og_tly
                        x2 = og_brx
                        y2 = og_bry

                        best_answer = pr.max()
                        answer = dict[pr.argmax()]
                        bounding_box = orig[og_tlx:og_brx, og_tly:og_bry]
                        bounding_box = cv2.resize(bounding_box, (128, 128), interpolation=cv2.INTER_LINEAR)

            # U ovom delu se gleda baš -l do l okruženje predviđanja

            for d1 in range(-l, l, step):
                for d2 in range(-l, l, step):
                    tlx = max(top_left[0] + d1, 0)
                    brx = min(bottom_right[0] + d1, dimx - 1)
                    tly = max(top_left[1] + d2, 0)
                    bry = min(bottom_right[1] + d2, dimy - 1)

                    og_tlx = int(tlx / scale_percent)
                    og_tly = int(tly / scale_percent)
                    og_brx = int(brx / scale_percent)
                    og_bry = int(bry / scale_percent)

                    sample = orig[og_tlx:og_brx, og_tly:og_bry]
                    sample = cv2.resize(sample, (128, 128), interpolation=cv2.INTER_LINEAR)
                    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                    sample = np.array([sample])

                    pr = model.predict(sample, verbose=0)

                    if pr.max() > best_answer:
                        x1 = og_tlx
                        y1 = og_tly
                        x2 = og_brx
                        y2 = og_bry

                        best_answer = pr.max()
                        answer = dict[pr.argmax()]
                        bounding_box = orig[og_tlx:og_brx, og_tly:og_bry]
                        bounding_box = cv2.resize(bounding_box, (128, 128), interpolation=cv2.INTER_LINEAR)

            fridge[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = -bg_factor
            fridge += bg_factor

            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
            sample = np.array([sample])

            print(best_answer)

            # Predviđanje se odbacuje ako model nije makar 75% siguran u rezultat

            if best_answer > 0.75:
                print(answer)
            else:
                print("False Positive")

            cv2.imshow('bounding_box', bounding_box)
            cv2.waitKey(0)

            backup[x1:x1+4, y1:y2] = 255
            backup[x1:x2, y1:y1+4] = 255
            backup[x1:x2, y2:y2+4] = 255
            backup[x2:x2+4, y1:y2] = 255

        else:
            backup = cv2.resize(backup, (400, 650), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Final', backup)
            cv2.imwrite('Final.jpg', backup)
            cv2.waitKey(0)
            break


# Ova funkcija određuje najveću sumu niza i tu vrednost vraća da bi našla najveću sumu submatrice


def kadane(arr, start, finish, n):
    sum = 0
    max_sum = -999999999999
    i = 0
    finish[0] = -1
    local_start = 0

    while i < n and (i - local_start <= (dimx / 5)):
        sum += arr[i]

        if sum < 0:
            sum = 0
            local_start = i + 1

        elif sum > max_sum:
            max_sum = sum
            start[0] = local_start
            finish[0] = i

        i += 1

    return max_sum


# Poređenjem submatrica, bira onu sa najvećom sumom i predstavlja je kao sledećeg kandidata za predviđanje


def find_max_sum(M):
    global dimx, dimy

    max_sum = -999999999999
    final_left, final_right, final_top, final_bottom = None, None, None, None

    sum = 0
    start = [0]
    finish = [0]

    for left in range(0, dimy - int(dimy / 5), 2):
        temp = [0] * dimx

        for right in range(min(left, dimy), min(left + int(dimy / 3), dimy), 2):
            for i in range(0, dimx, 1):
                temp[i] += M[i][right]

            if (right - left) >= int(dimy / 10):
                sum = kadane(temp, start, finish, dimx)
                if (1 + ((right - left) + (finish[0] - start[0])) / 100) != 0:
                    sum = sum / (1 + ((right - left) + (finish[0] - start[0])) / 100)

            if sum > max_sum and (right - left) > 5 and (finish[0] - start[0]) > 5:
                max_sum = sum
                final_left = left
                final_right = right
                final_top = start[0]
                final_bottom = finish[0]

    print("(Top, Left)", "(", final_top,
          final_left, ")")
    print("(Bottom, Right)", "(", final_bottom,
          final_right, ")")
    print("Max sum is:", max_sum)

    return final_top, final_left, final_bottom, final_right, max_sum

#"""

training_set = make_training_set()
validation_set = make_validation_set()
model = make_model(training_set, validation_set)

#"""

"""

#model1 = tf.keras.models.load_model("C:\\Users\\Oli\\PycharmProjects\\Fridger\\Models\\bestModel256.h5")
model2 = tf.keras.models.load_model("C:\\Users\\Oli\\PycharmProjects\\Fridger\\Models\\bestModel128.h5")
test_case(24, model2)
#test_case(30, model2)

"""


"""

test_img = cv2.imread('C:\\Users\\Oli\\Pictures\\Shrimp.png')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = np.array([test_img])

pr = model.predict(test_img)
prediction = pr.argmax()
print(pr)
print(prediction)

"""