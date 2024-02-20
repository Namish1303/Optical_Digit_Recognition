

from collections import defaultdict

train_data = [list(map(int, line.strip().split(","))) for line in open("train_TwoFifty.csv").read().strip().split("\n")]
train_data, validate_data = train_data[:50000], train_data[50000:]


def normalize(probs):
    sum_probs = sum(probs)
    #print(sum_probs)
    if sum_probs == 0:
        return normalize([1 for each in probs])
    eachProb = []
    for each in probs:
        eachProb.append(each/sum_probs)
    #print(len(probs))
   # print(len(eachProb))
    #print(eachProb[3])
    return eachProb

def normalize_dict(d):
    total = sum(d.values())
    for each in d:
        d[each] = d[each] / total
    #print(d) // probabilities of each number occuring in the train file
    return d

def train():
    # returns a pixel_prob
    prior_prob = defaultdict(int)
    counter = {i: {x: [0, 0] for x in range(10)} for i in range(784)}
    #print(counter)
    for data in train_data:
        data = convert_data(data)
        label = data[0]
        pixels = data[1:]
        prior_prob[label] += 1   ##increasing the occurences of each number as we traverse through the training array
        #print(prior_prob)
        for i in range(len(pixels)):
            counter[i][label][pixels[i]] += 1
            #print(counter[i][label][pixels[i]])
    #print(counter[284][4])
    #print(counter)
    prior_prob = normalize_dict(prior_prob)
    #at this point we have the prior probabilites of each number occuring in the training file
    for i in range(784):
        for x in range(10):
            counter[i][x] = normalize(counter[i][x])
    #print(counter[384][3])
    return prior_prob, counter

def argmax(lisProb):
    maximum = 0
    result = 0
    for i in range(len(lisProb)):
        if lisProb[i] > maximum:
            maximum = lisProb[i]
            result = i
        #print(maximum)    
    return result

# 1D array -> 784 (28 * 28) elements
def predict(data):
    
    global pixel_prob, prior_prob
    #print(pixel_prob)
    probs = [prior_prob[x] for x in range(10)]
    data = convert_data(data)
    for i in range(len(data)):
        for x in range(10):
            probs[x] *= pixel_prob[i][x][data[i]]
            #print(probs[x])
        probs = normalize(probs)
    print(probs)
    return argmax(probs)

def convert_data(data):
    #print([data[0]] + [1 if each > 0 else 0 for each in data[1:]])
    #converting the array of input into 1s and 0s like the assignment 2
    return [data[0]] + [1 if each > 0 else 0 for each in data[1:]]

def convert_1d_2d(arr1d, col=28):
    result = []
    for stscreen_print in range(0, len(arr1d), col):
        result.append(arr1d[stscreen_print:stscreen_print+col])
    return result

def show_image(pixels):
    pixels = convert_1d_2d(pixels)
    screen_print = ''

    for row in pixels:
        sing_row = ''
        for item in row:
                if(item == 0):
                    #print("HERE")
                    sing_row += " "
                else:
                    sing_row += "*"
                   # print(sing_row)
        
        screen_print += sing_row
        screen_print += "\n"
    #screen_print = "\n".join(["".join([ASCII[item//MAGIC] for item in row]) for row in pixels])W
    print(screen_print)

def run_test():

    data = [list(map(int, line.strip().split(","))) for line in open("test_fifteen.csv").read().strip().split("\n")]
    right = 0

    for i in range(len(data)):
        row = data[i]
        label = row[0]
        pixels = row[1:]
        prediction = predict(pixels)

        if label==prediction:
            right+=1

        print("NO.{}\npredict: {}\nactual: {}\naccuracy: {}".format(i, prediction, label, right/(i+1)))
        show_image(pixels)
        input("press Enter to continue")

if __name__=="__main__":
    prior_prob, pixel_prob = train()
    run_test()