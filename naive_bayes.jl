
path_train_positive = "C:\\Users\\deniz tekalp\\Desktop\\aclImdb\\train\\pos"
path_train_negative = "C:\\Users\\deniz tekalp\\Desktop\\aclImdb\\train\\neg"
path_test_positive = "C:\\Users\\deniz tekalp\\Desktop\\aclImdb\\test\\pos"
path_test_negative = "C:\\Users\\deniz tekalp\\Desktop\\aclImdb\\test\\neg"

files_train_positive = readdir(path_train_positive)
files_train_negative = readdir(path_train_negative)
files_test_positive = readdir(path_test_positive)
files_test_negative = readdir(path_test_negative)

xtrn_strings = []
for file in files_train_positive
    tmppath = string(path_train_positive,"\\", file)
    tmpfile = open(tmppath, "r")
    #push!(xtrn_strings, readlines(tmpfile)[1])
    x = readlines(tmpfile)[1]
    push!(xtrn_strings, x)
    close(tmpfile)
end

for file in files_train_negative
    tmppath = string(path_train_negative,"\\", file)
    tmpfile = open(tmppath, "r")
    #push!(xtrn_strings, readlines(tmpfile)[1])
    x = readlines(tmpfile)[1]
    push!(xtrn_strings, x)
    close(tmpfile)
end

xtst_strings = []
for file in files_test_positive
    tmppath = string(path_test_positive,"\\", file)
    tmpfile = open(tmppath, "r")
    x = readlines(tmpfile)[1]
    push!(xtst_strings, x)
    close(tmpfile)
end

for file in files_test_negative
    tmppath = string(path_test_negative,"\\", file)
    tmpfile = open(tmppath, "r")
    push!(xtst_strings, readlines(tmpfile)[1])
    close(tmpfile)
end

dict = Dict()
w2i(x) = get!(dict, x, 1+length(dict))
UNK = w2i("<unk>")

xtrn = []
for sentence in xtrn_strings
    tmparr = split(strip(lowercase(sentence)))
    tmparr=strip.(tmparr, [','])
    tmparr=strip.(tmparr, ['!'])
    tmparr=strip.(tmparr, ['.'])
    tmparr=strip.(tmparr, ['?'])
    tmparr=strip.(tmparr, [':'])
    tmparr=strip.(tmparr, [';'])
    tmparr=strip.(tmparr, ['*'])
    tmparr=strip.(tmparr, ['\"'])
    tmparr=strip.(tmparr, ['\\'])
    tmparr=strip.(tmparr, ['\"'])
    tmparr=strip.(tmparr, [')'])
    tmparr=strip.(tmparr, ['-'])
    tmparr=strip.(tmparr, ['\''])
    tmparr=strip.(tmparr, ['('])
    tmparr=strip.(tmparr, [')'])
    tmparr=strip.(tmparr, ['.'])
    w2i.(tmparr)
    push!(xtrn, tmparr)
end

xtst = []
for sentence in xtst_strings
    tmparr = split(strip(lowercase(sentence)))
    tmparr=strip.(tmparr, [','])
    tmparr=strip.(tmparr, ['!'])
    tmparr=strip.(tmparr, ['.'])
    tmparr=strip.(tmparr, ['?'])
    tmparr=strip.(tmparr, [':'])
    tmparr=strip.(tmparr, [';'])
    tmparr=strip.(tmparr, ['*'])
    tmparr=strip.(tmparr, ['\"'])
    tmparr=strip.(tmparr, ['\\'])
    tmparr=strip.(tmparr, ['\"'])
    tmparr=strip.(tmparr, [')'])
    tmparr=strip.(tmparr, ['-'])
    tmparr=strip.(tmparr, ['\''])
    tmparr=strip.(tmparr, ['('])
    tmparr=strip.(tmparr, [')'])
    tmparr=strip.(tmparr, ['.'])
    push!(xtst, tmparr)
end

w2i(x) = get(dict, x, UNK)

for i in 1:length(xtrn)
    xtrn[i] = w2i.(xtrn[i])
    xtst[i] = w2i.(xtst[i])
end

ytrn = ones(25000)
ytrn[1:12500] .= 2
ytst = ones(25000)
ytst[1:12500] .= 2

prior_prob_positive = sum(ytrn .-1)/length(ytrn)
prior_prob_negative = 1 - prior_prob_positive

negative_counts = ones(length(dict)) .+ 4
positive_counts = ones(length(dict)) .+ 4

for i in 1:length(xtrn)
    for word in xtrn[i]
        if ytrn[i] == 1  #if sentiment is negative
            negative_counts[word] += 1
        else           
            positive_counts[word]  += 1
        end
    end
end

negative_word_probs = negative_counts ./ sum(negative_counts)
positive_word_probs = positive_counts ./ sum(positive_counts)

function predict(sentence)
    negative_prob = log(prior_prob_negative)
    positive_prob = log(prior_prob_positive)
    for word in sentence
        negative_prob += log(negative_word_probs[word])
        positive_prob += log(positive_word_probs[word])
    end
    if negative_prob > positive_prob return 1
        else return 2
    end
end

print("accuracy: ", count(predict.(xtst) .== ytst) / length(xtst))


