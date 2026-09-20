import torch

with open("input.txt", "r", encoding='utf-8') as file:
    text = file.read()

length = len(text)
print("length of the dataset is ", length)

sample = text[0:500]
print(sample)

chars = sorted(set(text))
vocab_string = "".join(chars)
vocab_size = len(chars)

# print(vocab_string)
# print(vocab_size)

stoi = {}
itos = {}

for i, ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch

# print(stoi["a"])
# print(itos[39])
# print(stoi["\n"])

def encode(s):
    list1 = []
    for ch in s:
        list1.append(stoi[ch])
    return list1

def decode(i):
    list2 = []
    for no in i:
        list2.append(itos[no])
    return "".join(list2)

print(encode("hii there"))
print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype = torch.long)

print(data.shape)
print(data.dtype)
print(data[:100])

n = int(90/100 * len(data))
train_data = data[:n]
val_data = data[n:]

print(train_data.shape)
print(val_data.shape)

block_size = 8

x = train_data[:8]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context.tolist()} then target is {target}")

batch_size = 4

def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = val_data

    randomPosition = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in randomPosition])
    y = torch.stack([data[i+1 :i+block_size+1 ] for i in randomPosition])

    return x,y

xb, yb = get_batch("train")
print(xb.shape)
print(xb)
print(yb.shape)
print(yb)