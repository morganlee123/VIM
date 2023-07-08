import os
from re import split
from morphgen_demo_functions_skp2201 import run_DeepTalk_demo as DeepTalk
import soundfile as sf
from pathlib import Path

## This Demo file requires the trained_models directory to be present in the same location as this file

current_path=os.getcwd()
current_path=current_path.split('/')
pair=current_path[-1].split('_')[-1]
#print(pair)
#syn_model_dir =  Path('trained_models/'+pair+'/Synthesizer/logs-model_GST_ft/taco_pretrained')
#print('/research/iprobe-tmp/panisush/workspace/LibriSpeech/top_100_pairs/morph_source/'+pair)





## Set the current working directory to the path where 'DeepTalk-Deployment' directory is placed in your system
#os.chdir('/research/iprobe-tmp/panisush/workspace/interspeech_2022/DeepTalk_098_3467-466')


print(os.getcwd())
enc_module_name = "model_GST"  ## DO NOT CHANGE!


## Uncomment the block corresponding to the identity that you want to generate synthetic audios for

# ## For Generic
enc_model_fpath =  Path('../trained_models/Generic/Encoder/model_GST.pt')
#syn_model_dir =  Path('../trained_models/Generic/Synthesizer/logs-model_GST/taco_pretrained')
voc_model_fpath =  Path('../trained_models/Generic/Vocoder/model_GST/model_GST.pt')

#Desired model
#enc_model_fpath =  Path('trained_models/2288-1097/Encoder/model_GST.pt')
syn_model_dir =  Path('trained_models/'+pair+'/Synthesizer/logs-model_GST_ft/taco_pretrained')
#voc_model_fpath =  Path('trained_models/3467-466/Vocoder/model_GST_ft/model_GST_ft.pt')

#Reference Source Audio files of target speaker
ref_audio_path='/research/iprobe-tmp/panisush/workspace/LibriSpeech/top_100_pairs/morph_source/'+pair
gen_path=ref_audio_path+'/'+'gen'
isExist = os.path.exists(gen_path)
if not isExist:
    os.mkdir(gen_path)

## This is where the generated synthetic audio file is saved
generated_audio_path =ref_audio_path

## Edit the target_text to anything you want to be spoken out by the synthetic voice
target_text ='Bio metrics is the science of recognizing individuals based on their physical or behavioral traits.'



def genimp_pair(data_directory):
    test_list = []
    for root, dirs, files in os.walk(data_directory):
        files = filter(lambda f: f.endswith(('.wav','.WAV','.flac')), files)
        for file in files:
            #append the file name to the list
            test_list.append(os.path.join(root,file))
    
    #all_possible_unique_pair
    res = [(a, b) for idx, a in enumerate(test_list) for b in test_list[idx + 1:]]
    
    imposter=[]
    genuine=[]
    for pair in res:
        if((pair[0].split('/')[-2])==(pair[1].split('/')[-2])):
            genuine.append(pair)
        else:
            imposter.append(pair)
    return imposter

if os.path.isdir(ref_audio_path):
    filepair_list=genimp_pair(ref_audio_path)
    for sample_filepair in filepair_list:
        generated_audio_file=generated_audio_path+'/gen/' 
        
        file1=sample_filepair[0].split('.')[0]
        file2=sample_filepair[1].split('.')[0]
        file1=file1.split('/')[-1]
        file2=file2.split('/')[-1]
        
        ref_audio_path=sample_filepair
        generated_audio_file=generated_audio_file+file1+'_'+file2+'.wav'
        
        synthesized_wav, sample_rate, _ = DeepTalk(ref_audio_path=ref_audio_path, output_text=target_text,
                 enc_model_fpath=enc_model_fpath, enc_module_name=enc_module_name,
               syn_model_dir=syn_model_dir, voc_model_fpath=voc_model_fpath)

        # ## Write output to file
        sf.write(generated_audio_file, synthesized_wav, sample_rate, 'PCM_24')

#single file    
else:
    generated_audio_file=generated_audio_path+'_'+ref_audio_path.split('/')[1]
    generated_audio_file=generated_audio_file.replace('.WAV', '.wav')
    ## Run DeepTalk
    synthesized_wav, sample_rate, _ = DeepTalk(ref_audio_path=ref_audio_path, output_text=target_text,
            enc_model_fpath=enc_model_fpath, enc_module_name=enc_module_name,
            syn_model_dir=syn_model_dir, voc_model_fpath=voc_model_fpath)

    ## Write output to file
    sf.write(generated_audio_file, synthesized_wav, sample_rate, 'PCM_24')

