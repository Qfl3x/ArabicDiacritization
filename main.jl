using Flux:LSTM,functor,Dense,Chain,params,ADAM,softmax,onehotbatch,DataLoader,gradient,update!,OneHotArray,gpu,onecold
using Statistics: mean
using Base:size,show
using Flux.Losses:logitcrossentropy
using NNlib:gather
#using CUDA
using BSON:@load
import Flux.functor
import Base.size,Base.show

include("model.jl")


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

    input = replace(input,r"[[:punct:]]"=>"")
    input = replace(input,r"[\u060c]"=>"")

    input = replace(input, "\u200f" => "")
    input = replace(input,r"[0-9]"=>"")
    input = replace(input, r"[  ][  ]+" => " ")
    input = lstrip(input)
    input = rstrip(input)

    text = ""
    diac = zeros(Int, length(input)-length(collect(eachmatch(d_re,input))))

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



display(un_diacritize("قٌيِثً :"))
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

if !(@isdefined token_dict)
    token_dict = token_dict_constructor()
    un_token_dict = un_tokenize_dict_constructor()
end

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

#Construct dataset, don't run if already defined

function pad_input_vector(vec,i,type)
    return append!(vec,ones(type,i-length(vec)))
end

function pad_output_vector(vec,i,type)
    return append!(vec,zeros(type,i-length(vec)))
end

function returned_lines(train, trunc)
    lines = 0
    for line in train[1]
        line_over = false
        ind = 0
        while !line_over
            if length(line) - ind <= trunc
                lines += 1
                break
            end
            #println("ASD")
            ind = ind + 100
            lines += 1
        end
    end
    return lines
end

function data_trunc(train,trunc)
    space_ind = findall(x->x==1,train[1][1])
    space_ind = filter(x->x<=trunc,space_ind)
    chosen_ind = maximum(space_ind)

    length_of_data = returned_lines(train,trunc)
    train_x = ones(Int,trunc,length_of_data)
    train_y = zeros(Int,23,trunc,length_of_data)

    N_ind = 1
    for i in 1:length(train[1])
        curr_ind = 1
        while curr_ind < length(train[1][i])
            if length(train[1][i])<=trunc
                train_x[1:length(train[1][i]),N_ind] = train[1][i]
                train_y[:,1:length(train[2][i]),N_ind] = onehotbatch(train[2][i][1:end],0:22)

                break
            end
            space_ind = findall(x->x==1,train[1][i])
            space_ind = append!(space_ind,length(train[1][i]))
            space_ind = filter(x->x>=curr_ind,space_ind)
            #display(space_ind)
            space_ind = filter(x->x<=(trunc+curr_ind-1),space_ind)
            #display(space_ind)
            #display(length(train[1][i]))
            chosen_ind = maximum(space_ind)
            train_x[1:(chosen_ind-curr_ind+1),N_ind] = train[1][i][curr_ind:chosen_ind]
            train_y[:,1:(chosen_ind-curr_ind+1),N_ind] = onehotbatch(train[2][i][curr_ind:chosen_ind],0:22)

            curr_ind = chosen_ind + 1
       end
    end
    return train_x,train_y
end

#if !(@isdefined train)

    train_s = open("train.txt") do file
        read(file, String)
    end
    train_s = split(train_s, "\n")[1:end-1]

    val_s = open("val.txt") do file
        read(file, String)
    end
    val_s = split(val_s, "\n")[1:end-1]

train = data_trunc(partition_data(train_s[1:1000]), 300)
train_pre = partition_data(train_s[1:1000])
    #val = data_trunc(partition_data(val_s), 300)
#end
train_loader = DataLoader((data=train[1],label=train[2]),batchsize=128,shuffle=true)
#val_loader   = DataLoader((data=val[1],label=val[2]),batchsize=128,shuffle=true)

display(model)
loss(x,y) = logitcrossentropy(model(x),y)

opt = ADAM(0.005)
function train_model(epochs,model)
    model = model |> gpu

    ps = params(model)
    loss(x,y) = logitcrossentropy(model(x),y)
    opt = ADAM(0.005)

    train_error_arr = []
    val_error_arr = []
    train_error_current = 0.
    val_error_current   =0.

    for epoch in 1:epochs
        prev_ps = copy(ps)
        for (x,y) in train_loader
            x, y = x |> gpu, y |> gpu
            println(typeof(x))
            gs = gradient(() -> loss(x,y), ps)
            Flux.update!(opt, ps, gs)
            train_error_current += loss(x,y)
        end
        for (x,y) in val_loader
            x = gpu(x)
            y = gpu(y)
            val_error_current += loss(x,y)
        end


        train_error_current /= length(train_loader)
        val_error_current /= length(val_loader)

        @info "Epoch :" epoch
        @info "Training_Error :" train_error_current
        @info "Validation Error:" val_error_current
        train_error_arr = append!(train_error_arr,train_error_current)
        val_error_arr = append!(val_error_arr,val_error_current)
        if (epoch > 1) && ((val_error_current > val_error_arr[end-1]) || (isapprox(val_error_current,val_error_arr[end-1])))
            @info "Validation error capped"
            params(model) = prev_ps
            break
        end

        train_error_current = 0.
        val_error_current =0.
    end
    return cpu(model),train_error_arr,val_error_arr
end

#train_model(10)

#@load "model2.bson" model

accuracy(x,y) = mean(onecold(model(x)) .== onecold(y))
