
using Flux
using Flux:Recur,LSTMCell,@functor,Embedding
using NNlib:gather
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
@functor Embed

abstract type AbstractBLSTM{F} end
# forward  :: Recur{LSTMCell{Matrix{Float32}, Vector{Float32}, Tuple{Matrix{Float32}, Matrix{Float32}}}, Tuple{Matrix{Float32}, Matrix{Float32}}}
    # backward :: Recur{LSTMCell{Matrix{Float32}, Vector{Float32}, Tuple{Matrix{Float32}, Matrix{Float32}}}, Tuple{Matrix{Float32}, Matrix{Float32}}}

struct BLSTM{A,B,C}
    forward  :: Recur{LSTMCell{A,B,C},C}
    backward :: Recur{LSTMCell{A,B,C},C}
    outdim   :: Int
end


function BLSTM(in::Int,out::Int)
    forward = LSTM(in,out)
    backward = LSTM(in,out)
    return BLSTM(forward,backward,out*2)
end

function (m::BLSTM)(x::AbstractArray)
    forward_out  = m.forward(x)
    backward_out = reverse(m.backward(reverse(x,dims=3)),dims=3)
    return cat(forward_out,backward_out,dims=1)
end
#Flux.trainable(m::BLSTM) = (m.forward,m.backward)
@functor BLSTM

model = Chain(Embedding(37,100),
              BLSTM(100,50),
              Dense(100,100),
              # BLSTM(100,50),
              # Dense(100,100),
              # BLSTM(100,50),
              # Dense(100,100),
              # BLSTM(100,50),
              # Dense(100,100),
              Dropout(0.1),
              Dense(100,17))

function test_func()
    counter = 1
    opt = ADAM(0.05)
    loss(x,y) = logitcrossentropy(model(x),y)
    ps = params(model)
for (x,y) in train_loader


    x_train = Array{Int}(x')
    y_train = permutedims(y,[1,3,2])


    display(size(model(Array{Int}(x'))))
    display(size(permutedims(y,[1,3,2])))
    gs = gradient(() -> loss(x_train,y_train), ps)
            Flux.update!(opt, ps, gs)

    counter += 1
end
end
test_func()
