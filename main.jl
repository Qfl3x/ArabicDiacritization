using Flux:LSTM,functor,Dense,Chain,params,ADAM,softmax,onehotbatch,DataLoader,gradient,update!,OneHotArray
using Base:size,show
using Flux.Losses:logitcrossentropy
using NNlib:gather
import Flux.functor
import Base.size,Base.show

#Embedding layer: (From Transformers.jl)
#
"""
    Embed(size::Int, vocab_size::Int)
The Embedding Layer, `size` is the hidden size. `vocab_size` is the number of the vocabulary. Just a wrapper for embedding matrix.
"""
abstract type AbstractEmbed{F} end
struct Embed{F ,W <: AbstractArray{F}} <: AbstractEmbed{F}
    scale::F
    embedding::W
end

functor(e::Embed) = (e.embedding,), m -> Embed(e.scale, m...)

size(e::Embed, s...) = size(e.embedding, s...)

Embed(size::Int, vocab_size::Int; scale = one(Float32)) = Embed(Float32(scale), randn(Float32, size, vocab_size))

function (e::Embed)(x::AbstractArray{Int})
    if isone(e.scale)
        gather(e.embedding, x)
    else
        e(x, e.scale)
    end
end

(e::Embed{F})(x, scale) where {F} = gather(e.embedding, x) .* convert(F, scale)
(e::Embed)(x::Vector{Vector{Int64}}) = (e::Embed).(x)

show(io::IO, e::Embed) = print(io, "Embed($(size(e.embedding, 1)))")




struct DiacritizedText
    text::String
    diacritization::Vector{Int}
end
#Diacritazation Hashing ===============
# diac_dict = Dict{Char,Int}(
#     '\u064e' => 1,
#     '\u064f' => 2,
#     '\u0650' => 3,
#     '\u0652' => 4,
#     '\u064b' => 5,
#     '\u064c' => 6,
#     '\u064d' => 7,
#     '\u0651' => 20,
# )
d_without_shadda = "\u064b\u064c\u064d\u064e\u064f\u0650\u0652"

function construct_diac_dict()
    diac_dict=Dict{String,Int}()
    diac_dict[""] = 0
    counter = 1
    for diac in d_without_shadda
        diac_dict[string(diac)] = counter
        counter += 1
    end
    for diac in d_without_shadda
        diac_dict["\u0651"*diac] = counter
        diac_dict[diac*"\u0651"] = counter
        counter += 1
    end
    diac_dict["\u0651"] = counter
    return diac_dict

end
diac_dict = construct_diac_dict()

function hash_diac(diac::String)
    hash = 0
    for char in diac
        hash += diac_dict[char]
    end
    return hash
end


d = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"
d_re = r"[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652]"

function normalize_med(input)

    capturing_regex_w = r"[و](?![\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652])"
    capturing_regex_y = r"[ي](?![\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652])"
    capturing_regex_a = r"[ا](?![\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652])"
    capturing_regex_a_m = r"[ى](?![\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652])"

    input = replace(input,capturing_regex_a => "اَ")
    input = replace(input,capturing_regex_w => "اُ")
    input = replace(input,capturing_regex_y => "اِ")
    input = replace(input,capturing_regex_a_m => "اْ")
    return input
end
    
function un_diacritize(input)

    #input = normalize_med(input) #Normalize the مد
    #
    input = replace(input,r"[[:punct:]]"=>"")
    input = replace(input,r"[\u060c]"=>"")

    input = replace(input, "\u200f" => "")
    input = replace(input,r"[0-9]"=>"")
    input = replace(input, r"[  ][  ]+" => " ")
    input = lstrip(input)
    input = rstrip(input)

    text = ""
    diac = zeros(Int, length(input)-length(collect(eachmatch(d_re,input))))
    #diac = [Int(0) for _ = 1:(length(collect(eachmatch(r"[أ-ي ]",input)))+length(collect(eachmatch(r"[ئءؤةآ]",input))))]

    input = split(input, "")
    diac_count = 0
    diac_str = ""
    for char in input
        if !occursin(char, d)
            if diac_str != ""
                diac[diac_count] = diac_dict[diac_str]
            end
            text *= char

            diac_count += 1
            diac_str = ""
        else
            diac_str *= char
            #diac[diac_count] += diac_dict[only(char)]
        end
    end
    if diac_str != ""
        diac[diac_count] = diac_dict[diac_str]
    end

    return DiacritizedText(text,diac)
end

function diacritize(input)
    output = ""
    text = split(input.text, "")
    for i=1:length(input.text)
        output *= text[i]
        output *= input.diacritization[i]
    end
    output
end


s = open("train.txt") do file
    read(file, String)
end

s = split(s, "\n")[1:end-1]
display(un_diacritize("قٌيِثً :"))
# training_text = [training_data[i].text for i = 1:length(training_data)]
# training_diac = [training_data[i].diacritization for i = 1:length(training_data)]

token_str = " \u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062a\u062b\u062c\u062d\u062e\u062f\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063a\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064a"

function token_dict_constructor()
    D = Dict{Char,Int}()

    counter = Int(1)
    for char in token_str
        D[char] = counter
        counter += Int(1)
    end
    return D
end

function un_tokenize_dict_constructor()
    D = Dict{Int,Char}()

    counter = Int(1)
    for char in token_str
        D[counter] = char
        counter += Int(1)
    end
    return D
end

#if !(@isdefined token_dict)
    token_dict = token_dict_constructor()
    un_token_dict = un_tokenize_dict_constructor()
#end

function un_tokenize(word::Vector{Int})

    str = ""
    for letter in word
        str *= un_token_dict[letter]
    end
    return str
end


function retype_data(line)

    line_int = Array{Int,1}([token_dict[only(char)] for char in line])

    return line_int
end

function partition_data(training_data)
    training_data = un_diacritize.(training_data)
    training_text = [retype_data(training_data[i].text) for i = 1:length(training_data)]
    training_diac_in = [training_data[i].diacritization for i = 1:length(training_data)]
    #training_diac = Vector{Vector{Int}}([])

    # for line in 1:length(training_text)
    #     cursor = 1
    #     curr_arr = Vector{Vector{Int}}([])
    #     for word in training_text[line]
    #         curr_arr = append!(curr_arr, [training_diac_in[line][cursor:cursor+length(word)-1]])
    #         cursor  += length(word)
    #     end
    #     training_diac = append!(training_diac, [curr_arr])
    # end
    return training_text, training_diac_in
end

function partition_data_test(input)
    diac = Vector{Vector{Int}}([])
    text = retype_data(input.text)
    cursor = 1
    for word in text
        diac = append!(diac, [input.diacritization[cursor:cursor+length(word)-1]])
        cursor += length(word)
    end
    return text, diac
end

#display(partition_data_test(un_diacritize("قَامَ عُمَرٌ بِالسْلَامِ")))
#Construct dataset, don't run if already defined

function pad_input_vector(vec,i,type)
    return append!(vec,ones(type,i-length(vec)))
end

function pad_output_vector(vec,i,type)
    return append!(vec,zeros(type,i-length(vec)))
end
function format_output(train_y)
    #output_arr = OneHotArray{UInt32,23,2,3,Matrix{UInt32}}(undef,0,0,0)
    output_arr = reshape(onehotbatch(train_y[1],0:22),23,100,:)
    for line in train_y[2:end]
        output_arr =cat(output_arr,onehotbatch(line,0:22), dims=3) #push!(output_arr,onehotbatch(line,0:22))
    end
    return output_arr
    #display(size(onehotbatch.(train_y,0:22)))
    #train_y = reshape(onehotbatch.(train_y,0:22),23,length(train_y))
end
function data_trunc(train,trunc)
    #display(train[1][1][1:97])
    train_x = reshape(pad_input_vector(train[1][1][1:97],trunc,Int),100,:)
    line_y = pad_output_vector(train[2][1][1:97],trunc,Int)
    train_y = reshape(onehotbatch(line_y,0:22),23,100,:)
    train[1][1] = deleteat!(train[1][1],1:97)
    train[2][1] = deleteat!(train[2][1],1:97)
    #train_x = Vector{Vector{Int64}}([])
    #train_x = reshape(pad_vector([1][1][1:end],trunc,Int),100,:)
    #train_y = Vector{Vector{Int}}([])
    for i in 1:length(train[1])
        while length(train[1][i])>0
            if length(train[1][i])<=trunc
                train_x = hcat(train_x, pad_input_vector(train[1][i][1:end],trunc,Int))
                train_y = cat(train_y,onehotbatch(pad_output_vector(train[2][i][1:end],trunc,Int),0:22),dims=3)
                #train_x = hcat(train_x, [pad_vector(train[1][i][1:end],trunc,Int)])
                #train_x = hcat(train_x, [pad_vector(train[1][i][1:end],trunc,Int)])
                # train_x = append!(train_x,[pad_input_vector(train[1][i][1:end],trunc,Int)])
                # train_y = append!(train_y,[pad_vector(train[2][i][1:end],trunc,Int)])
                break
            end
            space_ind = findall(x->x==1,train[1][i])
            space_ind = filter(x->x<=trunc,space_ind)
            chosen_ind = maximum(space_ind)
            train_x = hcat(train_x, pad_input_vector(train[1][i][1:chosen_ind],trunc,Int))
            train_y = cat(train_y,onehotbatch(pad_output_vector(train[2][i][1:chosen_ind],trunc,Int),0:22),dims=3)
            #train_y = append!(train_y,[pad_vector(train[2][i][1:chosen_ind],trunc,Int)])
            train[1][i] = deleteat!(train[1][i],1:chosen_ind)
            train[2][i]= deleteat!(train[2][i],1:chosen_ind)
        end
    end
    #train_y =cat(format_output(train_y)...,dims=3)
    #train_y = format_output(train_y)
    #train_x =cat(train_x...,dims=2)
    return train_x,train_y
end


if !(@isdefined train)
    #cut_data = partition_data(s)
#display(length(s))
    train = data_trunc(partition_data(s[1:10000]),100)
end
train_loader = DataLoader((data=train[1],label=train[2]),batchsize=128,shuffle=true)
function nn()
    return Chain(
        Embed(100,37),
        #LSTM(50,40),
        LSTM(100,50),
        Dense(50,30),
        #x -> permutedims(x,[2,1]),
        Dense(30,23),
        softmax)
end
model = nn()

loss(x,y) = logitcrossentropy(model(x),y)
ps = params(model)
opt = ADAM(0.005)
function train_model(epochs)
    train_error_current = 0.
    for epoch in 1:epochs
        for (x,y) in train_loader
            gs = gradient(() -> loss(x,y),ps) # Gradient with respect to ps
            train_error_current += loss(x,y)
            update!(opt,ps,gs)
        end
        train_error_current /= length(train_loader)
        @info "Epoch :" epoch
        @info "Training_Error :" train_error_current
        train_error_current = 0.
    end
    return model
end
train_model(10)