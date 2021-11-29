using PyCall
using StatsBase:countmap

camel_normalize = pyimport("camel_tools.utils.normalize")
camel_database = pyimport("camel_tools.morphology.database")
camel_analyzer = pyimport("camel_tools.morphology.analyzer")

db = camel_database.MorphologyDB.builtin_db()
analyzer = camel_analyzer.Analyzer(db)




struct DiacritizedText
    text::String
    diacritization::Vector{Int8}
end
#Diacritazation Hashing ===============
diac_dict = Dict{Char,Int8}(
    '\u064e' => 1,
    '\u064f' => 2,
    '\u0650' => 3,
    '\u0652' => 4,
    '\u064b' => 5,
    '\u064c' => 6,
    '\u064d' => 7,
    '\u0651' => 20,
)
reverse_diac_dict = Dict{Int8,Char}(

    1 => '\u064e',
    2 => '\u064f',
    3 => '\u0650',
    4 => '\u0652',
    5 => '\u064b',
    6 => '\u064c',
    7 => '\u064d',
    20 => '\u0651'
)
function hash_diac(diac::String)
    hash = 0
    for char in diac
        hash += diac_dict[char]
    end
    return hash
end


d = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"

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
    input = camel_normalize.normalize_unicode(input)
    #input = normalize_med(input) #Normalize the مد
    text = ""
    diac = [Int8(0) for _ = 1:(length(collect(eachmatch(r"[أ-ي]",input)))+length(collect(eachmatch(r"[ئءؤةآ]",input))))]

    input = split(input, "")
    diac_count = 0
    for char in input
        if (match(r"[[:punct:]0-9]",char) === nothing)
            if !occursin(char, d)
                # if char === "إ" || char === "ئ" || char === "ء" #Normalizing Hamza
                #     text *= "أ"
                # elseif char == "ة"                       #Normalizing ت
                #     text *= "ت"
                # else
                    text *= char
                #end

                if !(match(r"[أ-ي]",string(char)) === nothing)
                    diac_count += 1
                end
            else
                diac[diac_count] += diac_dict[only(char)]
            end
        end

    end

    return DiacritizedText(text,diac)
end

function diacritize(input)
    output = ""
    text = split(input.text, "")
    for i=1:length(input.text)
        output *= text[i]
        if input.diacritization[i]>20
            output *= reverse_diac_dict[20]
            output *= reverse_diac_dict[input.diacritization[i]-20]
            continue
        end
        if input.diacritization[i] == 0
            continue
        end

        output *= reverse_diac_dict[input.diacritization[i]]
    end
    output
end


s = open("train.txt") do file
    read(file, String)
end

s = split(s, "\n")[1:end-1]
display(un_diacritize("قٌيِثً :"))

#TODO: Reintegrate punctuation.
function cut_into_words(line)
    #line = replace(line, r"[[:punct:]0-9]" => "")
    line = replace(line, r"[  ][  ]+" => " ")
    line = lstrip(line)
    line = rstrip(line)
    line = replace(line, "\u200f" => "")
    line = split(line, " ")
    # for word in line
    #     line_int = append!(line_int,[token_dict[only(char)] for char in word])
    # end
    return line
end

asp = "c i p na"
cas = "n a g na u"
form_gen = "f m na"
form_num = "s d p na u"
gen = "f m na u"
mod = "i j s na u"
num = "s d p na u"
per = "1 2 3 na"
rat = "r i na n y u"
stt = "c d i na u"
vox = "a p na u"
pos  = "noun noun_prop noun_num noun_quant adj adj_comp adj_num adv adv_interrog adv_rel pron pron_dem pron_exclam pron_interrog pron_rel verb verb_pseudo part part_dem part_det part_focus part_fut part_interrog part_neg part_restrict part_verb part_voc prep abbrev punc conj conj_sub interj digit latin"
prc0 = "0 na Aa_prondem Al_det AlmA_neg lA_neb mA_neg ma_neg mA_part mA_rel"
prc1 = "0 na <i\$_interrog bi_part bi_prep bi_prog Ea_prep EalaY_prep fiy_prep hA_dem Ha_fut ka_prep la_emph la_prep la_rc libi_prep laHa_emphfut laHa_rcfut li_jus li_sub li_prep min_prep sa_fut ta_prep wa_part wa_prep wA_voc yA_voc"
prc2 = "0 na fa_conj fa_conn fa_rc fa_sub wa_conj wa_part wa_sub"
prc3 = "0 na >a_ques"
enc0 = "0 na 1s_dobj 1s_poss 1s_pron 1p_dobj 1p_poss 1p_pron 2d_dobj 2d_poss 2d_pron 2p_dobj 2p_poss 2p_pron 2fs_dobj 2fs_poss 2fs_pron 2fp_dobj 2fp_poss 2fp_pron 2ms_dobj 2ms_poss 2ms_pron 2mp_dobj 2mp_poss 2mp_pron 3d_dobj 3d_poss 3d_pron 3p_dobj 3p_poss 3p_pron 3fs_dobj 3fs_poss 3fs_pron 3fp_dobj 3fp_poss 3fp_pron 3ms_dobj 3ms_poss 3ms_pron 3mp_dobj 3mp_poss 3mp_pron Ah_voc lA_neg ma_interrog mA_interrog man_interrog ma_rel mA_rel man_rel ma_sub mA_sub"

function form_dictionaries()
    asp_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(asp," ")
        asp_dict[result] = c
        c+=1
    end

    cas_dict = Dict{String,Int}()
    c=Int(1)
    for reslt in split(cas, " ")
        cas_dict[reslt] = c
        c+=1
    end

    form_gen_dict = Dict{String,Int}()
    c=Int(1)
    for reslt in split(form_gen, " ")
        form_gen_dict[reslt] = c
        c+=1
    end

    form_num_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(form_num," ")
        form_num_dict[result] = c
        c+=1
    end

     gen_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(gen," ")
        gen_dict[result] = c
        c+=1
    end

    mod_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(mod," ")
        mod_dict[result] = c
        c+=1
    end

    num_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(num," ")
        num_dict[result] = c
        c+=1
    end

    per_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(per," ")
        per_dict[result] = c
        c+=1
    end

    rat_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(rat," ")
        rat_dict[result] = c
        c+=1
    end

    stt_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(stt," ")
        stt_dict[result] = c
        c+=1
    end

    vox_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(vox," ")
        vox_dict[result] = c
        c+=1
    end

    pos_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(pos," ")
        pos_dict[result] = c
        c+=1
    end

    prc0_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(prc0," ")
        prc0_dict[result] = c
        c+=1
    end

    prc1_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(prc1," ")
        prc1_dict[result] = c
        c+=1
    end

    prc2_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(prc2," ")
        prc2_dict[result] = c
        c+=1
    end

    prc3_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(prc3," ")
        prc3_dict[result] = c
        c+=1
    end

    enc0_dict = Dict{String,Int}()
    c = Int(1)
    for result in split(enc0," ")
        enc0_dict[result] = c
        c+=1
    end

    return asp_dict, cas_dict, form_gen_dict,form_num_dict,gen_dict, mod_dict, num_dict, per_dict, rat_dict, stt_dict, vox_dict, pos_dict, prc0_dict,prc1_dict,prc2_dict,prc3_dict,enc0_dict
end

asp_dict, cas_dict, form_gen_dict,form_num_dict,gen_dict, mod_dict, num_dict, per_dict, rat_dict, stt_dict, vox_dict, pos_dict, prc0_dict,prc1_dict,prc2_dict,prc3_dict,enc0_dict = form_dictionaries()


#!==============Data Acquisition Bloc ==================!#


function clean_data(training_data)
    training_text = [cut_into_words(training_data[i].text) for i = 1:length(training_data)]
    training_diac_in = [training_data[i].diacritization for i = 1:length(training_data)]
    training_diac = Vector{Vector{Vector{Int8}}}([])

    for line in 1:length(training_text)
        cursor = 1
        curr_arr = Vector{Vector{Int8}}([])
        for word in training_text[line]
            curr_arr = append!(curr_arr, [training_diac_in[line][cursor:cursor+length(word)-1]])
            cursor  += length(word)
        end
        training_diac = append!(training_diac, [curr_arr])
    end
    return training_text, training_diac
end
#display(clean_data_test(un_diacritize("قَامَ عُمَرٌ بِالسْلَامِ")))
#Construct dataset, don't run if already defined
if !(@isdefined train)
    training_data = un_diacritize.(s)
    train = clean_data(training_data)
end

#!===================================================!#

#stems = Vector{String}([])
#
property_list = split("asp cas form_gen form_num gen mod num per rat stt vox pos prc0 prc1 prc2 prc3 enc0"," ")
dict_list = [asp_dict,cas_dict,form_gen_dict,form_num_dict,gen_dict,mod_dict,num_dict,per_dict,rat_dict,stt_dict,vox_dict,pos_dict,prc0_dict,prc1_dict,prc2_dict,prc3_dict,enc0_dict]


function analyze_data(train)
    text_arr = Vector{Vector{Int}}([])
    diac_arr = Vector{Vector{Int8}}([])
    stems = Vector{String}([])
    rep_dict = Dict{Any,Int}()
    for i = 1:length(train[1])
        for j = 1:length(train[1][i])
            highest_prob = -100.
            most_likely_analysis = 0
            analyses = analyzer.analyze(train[1][i][j])
            for analysis in analyses
                prob = analysis["pos_lex_logprob"]
                if  prob > highest_prob
                    most_likely_analysis = analysis
                    highest_prob = prob
                end
            end
            analysis = most_likely_analysis
            # display(train[1][i][j])
            # display(length(analyses))
            # display((i,j))
            if analyses == [] #Out of Vocab.
                stems = append!(stems,[train[1][i][j]])
                word_arr = [length(train[1][i][j]),length(stems),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                text_arr = append!(text_arr, [word_arr])
                diac_arr = append!(diac_arr, [train[2][i][j]])
                continue
            end

            if !(analysis["root"] in stems)
                stems = append!(stems,[analysis["root"]])
                root_id = length(stems)
            else
                root_id = findfirst(x->x==analysis["root"],stems)
            end
            # display(asp_dict[analysis["asp"]])
            # display(cas_dict[analysis["cas"]])
            # display(form_gen_dict[analysis["form_gen"]])
            # display(form_num_dict[analysis["form_num"]])
            # #display(gen_dict[analysis["gen"]])
            # display(mod_dict[analysis["mod"]])
            # num_dict[analysis["num"]]
            # per_dict[analysis["per"]]
            # rat_dict[analysis["rat"]]
            # stt_dict[analysis["stt"]]
            # vox_dict[analysis["vox"]]
            # pos_dict[analysis["pos"]]
            # prc0_dict[analysis["prc0"]]
            # prc1_dict[analysis["prc1"]]
            # prc2_dict[analysis["prc2"]]
            # prc3_dict[analysis["prc3"]]
            # enc0_dict[analysis["enc0"]]
            word_arr = zeros(Int,19)
            word_arr[1] = length(train[1][i][j])
            word_arr[2] = root_id
            for i = 1:length(property_list)
                if property_list[i] in keys(analysis)
                    pre_dict = analysis[property_list[i]]
                    if pre_dict == "-"
                        continue
                    end

                    #println(property_list[i])
                    word_arr[2+i] = dict_list[i][pre_dict]
                end
            end
            word_arr_in_text_arr = false
            for ind in findall(x->x==word_arr,text_arr)
                if diac_arr[ind] == train[2][i][j]
                    word_arr_in_text_arr = true
                    rep_dict[[text_arr[ind],diac_arr[ind]]] += 1
                    break
                end
            end

            if !word_arr_in_text_arr
                text_arr = append!(text_arr,[word_arr])
                diac_arr = append!(diac_arr,[train[2][i][j]])
                rep_dict[[word_arr,train[2][i][j]]] = 1
            end

        end
    end
    return text_arr, diac_arr, stems, rep_dict
end
little_train = [train[1][1:2000],train[2][1:2000]]
#
function dumb_MLE(text_arr,diac_arr,rep_dict)
    seen_arrays = Vector{Vector{Int}}([])
    word_diac_dict = Dict{Vector{Int},Vector{Int8}}()
    for i = 1:length(text_arr)
        if text_arr[i] in seen_arrays
            continue
        end

        max_val = 0
        chosen = 0
        for key in keys(rep_dict)
            if key[1] == text_arr[i] && rep_dict[key]>max_val
                chosen = key[2]
                max_val = rep_dict[key]
            end
        end
        word_diac_dict[text_arr[i]] = diac_arr[i]
        seen_arrays = append!(seen_arrays,[text_arr[i]])
    end
    return word_diac_dict
end

#text_arr, diac_arr, stems, rep_dict = analyze_data(little_train)
#word_diac_dict = dumb_MLE(text_arr,diac_arr,rep_dict)
#

#!===========From word to diacritization=============!#
#
function diacritize_word(word,stems,word_diac_dict)
    highest_prob = -100.
    most_likely_analysis = 0
    analyses = analyzer.analyze(word)
    for analysis in analyses
        prob = analysis["pos_lex_logprob"]
        if  prob > highest_prob
            most_likely_analysis = analysis
            highest_prob = prob
        end
    end
    analysis = most_likely_analysis
    if analyses == [] #Out of Vocab.
        stem = findfirst(x->x==word,stems)
        word_arr = [word,stem,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        if stem === nothing
            return word
        end
        diac = word_diac_dict[word_arr]
        return diacritize(DiacritizedText(word,diac))
    end

    if !(analysis["root"] in stems)
        return word
    else
        root_id = findfirst(x->x==analysis["root"],stems)
    end

    word_arr = zeros(Int,19)
    word_arr[1] = length(word)
    word_arr[2] = root_id
    for i = 1:length(property_list)
        if property_list[i] in keys(analysis)
            pre_dict = analysis[property_list[i]]
            if pre_dict == "-"
                continue
            end

            #println(property_list[i])
            word_arr[2+i] = dict_list[i][pre_dict]
        end
    end
    diac = word_diac_dict[word_arr]
    display(diac)
    return diacritize(DiacritizedText(word,diac))
end
