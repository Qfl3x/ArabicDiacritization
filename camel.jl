#Diacritization project

#TODO: Cut this file into smaller files.

using PyCall:pyimport
using StatsBase:countmap

camel_normalize = pyimport("camel_tools.utils.normalize")       #Needed for normalizing unicode
camel_database = pyimport("camel_tools.morphology.database")    #Morphology database
camel_analyzer = pyimport("camel_tools.morphology.analyzer")    #Morphology analyzer

#Initializing morphology analyzer
db = camel_database.MorphologyDB.builtin_db()
analyzer = camel_analyzer.Analyzer(db)



#!========Diacritazation Hashing ========!#
function hash_diac(diac::String)    #Function used to map from the diacritics of A LETTER to corresponding Int8. Useful due to the Shadda situation.
    #
    diac_dict = Dict{Char,Int8}( #This is the diacritics lookup dictionary. 20 is the Shadda diacritic while others are all diacritics excluding Shadda.
    '\u064e' => 1,           #A letter can have a Shadda + ordinary diacritic (except Tanween, however considering Tanween as an ordinary diacritic does no harm).
    '\u064f' => 2,           #The goal is to go from a diacritic to an Int8. Saving memory and allowing the program to use ordinary Array operations especially indexing.
    '\u0650' => 3,           #This is necessary because in a unicode string, not all indexes correspond to a unicode character.
    '\u0652' => 4,           #(Notice how all loops through strings split the text first. Memory footprint should be minimal)
    '\u064b' => 5,           #For more information: https://docs.julialang.org/en/v1/manual/strings/#Unicode-and-UTF-8
    '\u064c' => 6,           #Note: 20 is arbitrary.
    '\u064d' => 7,
    '\u0651' => 20,
)

    hash = 0
    for char in diac
        hash += diac_dict[char]
    end
    return hash
end
"""A simple datastructure to hold the text and its diacritization. Has 2 fields:
    * text: the un-diacritized text
    * diacritization: The Int8 Vector holding the diacritics of each letter
"""
struct DiacritizedText
    text::String
    diacritization::Vector{Int8}
end
#!======================================!#

"""
Function for normalizing the types of vowel elongation (مد). Currently unused as it confuses the morphology analyzer.
Note that all normalizations are DISABLED (except unicode). They all seem to mess with the analyzer. (Which is odd considering CAMel recommends normalizing beforehand) <--!!
"""
function normalize_med(input) #

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

"""
Function that splits a text into undiacritized text and its diacritics. Output is a DiacritizedText datastructure.
Also eliminates numbers and punctuation. (Corpus is split into sentences anyways)
"""
function un_diacritize(input)

    d = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"  #String of diacritics for lookup below.


    input = camel_normalize.normalize_unicode(input)
    #input = normalize_med(input) #Normalize the مد
    text = ""
    diac = [Int8(0) for _ = 1:(length(collect(eachmatch(r"[أ-ي]",input)))+length(collect(eachmatch(r"[ئءؤةآ]",input))))]
    #Intializes diacritics array. It looks for characters and initializes an array of Int8 zeroes for each one. (Spaces not considered for memory)

    input = split(input, "") #Splitting input into its unicode characters.
    diac_count = 0
    for char in input
        if (match(r"[[:punct:]0-9]",char) === nothing) #Remove numbers and punctuation
            if !occursin(char, d) #char is not a diacritic
                #All normalizations are DISABLED
                # if char === "إ" || char === "ئ" || char === "ء" #Normalizing Hamza
                #     text *= "أ"
                # elseif char == "ة"                       #Normalizing ت
                #     text *= "ت"
                # else
                text *= char

                if !(match(r"[أ-ي]",string(char)) === nothing) #Diacritic counter necessary as some letters may have more than one and some have none.
                    diac_count += 1
                end
            else
                diac[diac_count] += diac_dict[only(char)] #picks up the diacritics counted. `only`` function is necessary as the dictionaries map Char to Int8, not strings.
            end
        end

    end

    return DiacritizedText(text,diac)
end

"""
Goes from DiacritizedText to a diacritized Arabic String. (Numbers and punctuation is not preserved if fed the output of `un_diacritize`)
"""
function diacritize(input)

    reverse_diac_dict = Dict{Int8,Char}(    #The reverse lookup dictionary for diacritics, used for going from the Int8 to diacritic.
                                            1 => '\u064e',
                                            2 => '\u064f',
                                            3 => '\u0650',
                                            4 => '\u0652',
                                            5 => '\u064b',
                                            6 => '\u064c',
                                            7 => '\u064d',
                                            20 => '\u0651'
                                            )

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

display(un_diacritize("قٌيِثً :")) #Sanity check

#TODO: Reintegrate punctuation.
function cut_into_words(line) #Straightforward function, splits a line into words.
    line = replace(line, r"[  ][  ]+" => " ")
    line = lstrip(line)
    line = rstrip(line)
    line = replace(line, "\u200f" => "") #Right-to-Left unicode hidden character. Present in all lines.
    line = split(line, " ")
    return line
end

#!==================== Morophological properties mapping =================!#
#This next large bloc is used to go from the output of the CAMel morphological analysis function to Int objects. This is to have simple Int vectors going into the model. The result is a bunch of lookup dictionaries
#Note issue: https://github.com/CAMeL-Lab/camel_tools/issues/74. The main source of these is the CAMel documentation: https://camel-tools.readthedocs.io/en/latest/reference/camel_morphology_features.html
#However, certain tags are missing from the documentation but are output by the analyzer.


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

#!=================Bloc over===========================!#


#!==============Data Acquisition Bloc==================!#

"""
Main data pipeline function. Cuts the lines into words, and forms the vector of diacritization vectors (Int8 Vectors).
"""
function clean_data(training_data)
    training_data = un_diacritize(training_data)    #un_diacritize the text
    training_text = [cut_into_words(training_data[i].text) for i = 1:length(training_data)]
    training_diac_in = [training_data[i].diacritization for i = 1:length(training_data)]
    training_diac = Vector{Vector{Vector{Int8}}}([])

    for line in 1:length(training_text)             #The loop is used to cut the large diacritization vectors of lines into each word's diacritization vector.
        cursor = 1                                  #Costs some CPU cycles but saves memory.
                                                    #Uses a cursor array to know where each word ends or begins. May be faster and more efficient to use pointers,
        curr_arr = Vector{Vector{Int8}}([])         #but since the function only runs once (doesn't run if the training_data is already cleaned) and doesn't take much time
        for word in training_text[line]             #it's not worth to optimize.
            curr_arr = append!(curr_arr, [training_diac_in[line][cursor:cursor+length(word)-1]])
            cursor  += length(word)
        end
        training_diac = append!(training_diac, [curr_arr])
    end
    return training_text, training_diac
end
if !(@isdefined train) #Construct dataset, don't run if already defined

    s = open("train.txt") do file #the training file from the dataset repo.
        read(file, String)
    end

    s = split(s, "\n")[1:end-1] #Cuts it into lines.

    train = clean_data(s)
end

#!===================================================!#

#These two arrays are used to loop through the properties. Necessary because sometimes the property is undefined (returns "-").
property_list = split("asp cas form_gen form_num gen mod num per rat stt vox pos prc0 prc1 prc2 prc3 enc0"," ")
dict_list = [asp_dict,cas_dict,form_gen_dict,form_num_dict,gen_dict,mod_dict,num_dict,per_dict,rat_dict,stt_dict,vox_dict,pos_dict,prc0_dict,prc1_dict,prc2_dict,prc3_dict,enc0_dict]

"""
The current function that goes from each word to its vector of properties.
For each word, the output is a 19-length Int Vector.
Words with the same analysis and diacritization are not repeated, instead a rep_dict is used, which maps
the pair [word analysis, diacritization] into it's current count.
Output should be ready to be fed to the classification model. (Maybe after dealing with the listed problems)

Doing the analysis of a word with CAMel's analyzer provides a Vector of possible analysis.
One of the important improvements currently is to make a smarter selection of the analysis then just the most common in CAMel's dataset.
Another is context, the output for each word is independent of what other words may have occured in the sentence
A third is out of vocab. As CAMel fails on certain not-so-common but easily recognized words (first example is :أعضاد. It is a simple plural form
of عضد which is correctly detected by CAMel while its plural isn't even though it follows the form أفعال a common plural form and not
a special case by any means)

(After doing a few tests on CAMel's own diacritization tool, it appears to suffer from the same issue)

"""
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

            #TODO: Something smarter than "just pick the one with the highest pos_lex_logprob"
            for analysis in analyses #Checks for highest pos_lex_logprob
                prob = analysis["pos_lex_logprob"]
                if  prob > highest_prob
                    most_likely_analysis = analysis
                    highest_prob = prob
                end
            end
            analysis = most_likely_analysis
            if analyses == [] #Out of Vocab. case
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
            #DEBUG BLOC necessary due to the discripancy between CAMel's documentation flags and the actual flags.
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
            for i = 1:length(property_list) #Goes through the properties one by one to check if they're undefined.

                if property_list[i] in keys(analysis) #Certain property keys ARE NOT in the analysis. This is necessary.
                    pre_dict = analysis[property_list[i]]
                    if pre_dict == "-" #Could add a dash to every property dictionary I suppose.
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

"""
Dumb MLE model. Just looks up the most common diacritization of each word analysis (which itself it the most common
analysis on CAMel's database). Does not take context into account, nor does it take word similarities
"""
function dumb_MLE(text_arr,diac_arr,rep_dict) #TODO: Better model
    seen_arrays = Vector{Vector{Int}}([]) #Array of seen words. as rep_dict includes certain pairs with the same word analysis but different diacritization. We only take the one most common.
    word_diac_dict = Dict{Vector{Int},Vector{Int8}}()
    for i = 1:length(text_arr)
        if text_arr[i] in seen_arrays #Pass seen words
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

little_train = [train[1][1:2000],train[2][1:2000]]  #This is the limit of constant work on my RAM (8GB)
text_arr, diac_arr, stems, rep_dict = analyze_data(little_train)
word_diac_dict = dumb_MLE(text_arr,diac_arr,rep_dict)
#

#!===========From word to diacritization=============!#
"""
Inference part. Goes from a word (String) to it's corresponding word_arr then checks for its existence in the analyzed words.
After finding it, the word is diacritized and output.

If the word is not found, it just outputs the word back.
"""
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
