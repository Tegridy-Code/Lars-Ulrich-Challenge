#!/usr/bin/env python
# coding: utf-8

# # Lars Ulrich Challenge (ver. 1.0)
# 
# ## "Music comes from the heart!" ---LU
# 
# ***
# 
# Powered by tegridy-tools TMIDIX Optimus Processors: https://github.com/Tegridy-Code/tegridy-tools
# 
# ***
# 
# Credit for GPT2-RGA code used in this colab goes out @ Sashmark97 https://github.com/Sashmark97/midigen and @ Damon Gwinn https://github.com/gwinndr/MusicTransformer-Pytorch
# 
# ***
# 
# WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/
# 
# ***
# 
# #### Project Los Angeles
# 
# #### Tegridy Code 2021
# 
# ***

# # Setup Environment

# ## Clone repo and install dependencies

# In[ ]:


get_ipython().system('git clone https://github.com/Tegridy-Code/Lars-Ulrich-Challenge')
    
get_ipython().system('pip install torch')
get_ipython().system('pip install tqdm')


# ## Import all needed modules

# In[ ]:


print('Loading needed modules. Please wait...')
import os
from datetime import datetime
import secrets
import copy
import tqdm
from tqdm import auto

os.chdir('Lars-Ulrich-Challenge')

print('Loading TMIDIX module...')
import TMIDIX
print('Loading GPT2RGA module...')
from GPT2RGA import *


# # Algorithmic Drums Matching

# ## Load Clean MIDI Pitches-Drums Database

# In[ ]:


notes, drums = TMIDIX.Tegridy_Any_Pickle_File_Reader('clean_midi_PDM')
print('Done!')


# ## Generate Drums from the Database

# In[ ]:


print('Lars Ulrich Challenge Algorithmic Drums Generator')
print('Project Los Angeles')
print('Tegridy Code 2021')

source_MIDI_file = 'Nothing Else Matters.mid'
pass_through_MIDI = True # Pass-through or not all MIDI events

# print('Loading MIDI file...')
mel_crd_f = []

score = TMIDIX.midi2score(open(source_MIDI_file, 'rb').read())

all_events_matrix = []
events_matrix = []
itrack = 1

while itrack < len(score):
    for event in score[itrack]:
        
        if event[0] == 'note' and event[3] != 9:
            events_matrix.append(event)
            
        if event[0] == 'note' and event[3] == 9:
            pass
        else:
            all_events_matrix.append(event)
        
    itrack += 1
    
# print('Grouping by start time. This will take a while...')
values = set(map(lambda x:x[1], events_matrix)) # Non-multithreaded function version just in case

groups = [[y for y in events_matrix if y[1]==x] for x in values] # Grouping notes into chords while discarting bad notes...

mel_crd = []    

# print('Sorting events...')
for items in groups:

    items.sort(reverse=True, key=lambda x: x[4]) # Sorting events by pitch

    mel_crd.append(items) # Creating final chords list

mel_crd.sort(reverse=False, key=lambda x: x[0][1])

mel_crd_f.extend(mel_crd)

src_pitches = []

for m in mel_crd_f:
    src = [y[4] for y in m]
    src_pitches.append(src)
    
final_song = []

drums_map = []

for i in tqdm(range(len(src_pitches))):
    try:
        idx = notes.index(src_pitches[i])
    except:
        idx = -1
        # print('not in list')
        pass
    final_song.extend(mel_crd_f[i])
    
    if idx != -1:
        for d in drums[idx]:
            # if d < 40:
            drum = [['note', mel_crd_f[i][0][1], mel_crd_f[i][0][2], 9, d, 100 ]]
            final_song.extend(drum)
            drums_map.extend(drum)
            
        idx = -1

if pass_through_MIDI:
    
    output =  all_events_matrix + drums_map
    
    output.sort(key=lambda x: x[1])
    
    midi_data = TMIDIX.score2midi([score[0], output,
                                    [['track_name', 0, bytes('Project Los Angeles', 'utf-8')]],
                                    [['track_name', 0, bytes('Lars Ulrich Challenge', 'utf-8')]]])

    detailed_MIDI_stats = TMIDIX.score2stats([score[0], output,
                                                [['track_name', 0, bytes('Project Los Angeles', 'utf-8')]],
                                                [['track_name', 0, bytes('Lars Ulrich Challenge', 'utf-8')]]])

    with open('./LUC-Algorithmic-Composition' + '.mid', 'wb') as midi_file:
        midi_file.write(midi_data)
        midi_file.close()
    
    print('Done! Enjoy! :)')
    
    print(detailed_MIDI_stats)
    
else:             
    TMIDIX.Tegridy_SONG_to_MIDI_Converter(final_song, 
                                          output_file_name='./LUC-Algorithmic-Composition',
                                          output_signature = 'Project Los Angeles', 
                                          track_name = 'Lars Ulrich Challenge',
                                          number_of_ticks_per_quarter=score[0],
                                          list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0])


# # Artificial Intelligence Drums Matching

# ## Unzip the Model

# In[ ]:


print('=' * 70)
print('Unzipping the model...Please wait...')
print('=' * 70)

get_ipython().system('cat Clean-MIDI-Transformer-Model.zip* > Clean-MIDI-Transformer-Model.zip')
print('=' * 70)

get_ipython().system('unzip -j Clean-MIDI-Transformer-Model.zip')
print('=' * 70)

print('Done! Enjoy! :)')
print('=' * 70)


# ## Load Clean MIDI Transformer Model

# In[ ]:


#@title Load/Reload the model
full_path_to_model_checkpoint = 'Clean-MIDI-Transformer-Model.pth'

print('Loading the model...')
config = GPTConfig(VOCAB_SIZE, 
                   max_seq,
                   dim_feedforward=dim_feedforward,
                   n_layer=6, 
                   n_head=8, 
                   n_embd=512,
                   enable_rpr=True,
                   er_len=max_seq)

model = GPT(config).to(get_device())

model.load_state_dict(torch.load(full_path_to_model_checkpoint))
model.eval()
print('Done!')


# ## Generate Drums from the Model

# In[ ]:


print('Lars Ulrich Challenge Model Generator')
print('Project Los Angeles')
print('Tegridy Code 2021')

source_MIDI_file = 'Nothing Else Matters.mid' # soutce MIDI file
memory_tokens = 64 # recommended 32-64
pass_through_MIDI = True # Pass-through or not all MIDI events

# print('Loading MIDI file...')

mel_crd_f = []
score = TMIDIX.midi2score(open(source_MIDI_file, 'rb').read())

all_events_matrix = []
events_matrix = []
itrack = 1

while itrack < len(score):
    for event in score[itrack]:
        
        if event[0] == 'note' and event[3] != 9: # reading all notes events except for the drums
            events_matrix.append(event)
            
        if event[0] == 'note' and event[3] == 9:
            pass
        else:
            all_events_matrix.append(event)
        
    itrack += 1
    
# print('Grouping by start time. This will take a while...')
values = set(map(lambda x:x[1], events_matrix)) # Non-multithreaded function version just in case

groups = [[y for y in events_matrix if y[1]==x] for x in values] # Grouping notes into chords while discarting bad notes...

mel_crd = []    

# print('Sorting events...')
for items in groups:

    items.sort(reverse=True, key=lambda x: x[4]) # Sorting events by pitch

    mel_crd.append(items) # Creating final chords list

mel_crd.sort(reverse=False, key=lambda x: x[0][1])

mel_crd_f.extend(mel_crd)

src_pitches = []

for m in mel_crd_f:
    src = [y[4] for y in m]
    src_pitches.append(src)
    
final_song = []
out = [0]

# print('Generating drums...')
drums_map = []

for i in tqdm(range(len(src_pitches))):
#for i in tqdm([0]):
    
    
    final_song.extend(mel_crd_f[i])
    if memory_tokens > 0:
        number_of_tokens_to_generate = memory_tokens+32
        data = out[-memory_tokens:] + [388] + src_pitches[i] + [128]
    else:
        number_of_tokens_to_generate = 32
        data = [388] + src_pitches[i] + [128]
        
    rand_seq = model.generate(torch.Tensor(data), target_seq_length=number_of_tokens_to_generate, stop_token=128)
    out = rand_seq[0].cpu().numpy().tolist()
    out_cc = out[out.index(128, len(data)-1)+1:]

    for d in out_cc:
        drum = [['note', mel_crd_f[i][0][1], mel_crd_f[i][0][2], 9, d, 100 ]]
        final_song.extend(drum)
        drums_map.extend(drum)
        
print('Writing MIDI...')

if pass_through_MIDI:
    
    output =  all_events_matrix + drums_map
    
    output.sort(key=lambda x: x[1])
    
    midi_data = TMIDIX.score2midi([score[0], output,
                                    [['track_name', 0, bytes('Project Los Angeles', 'utf-8')]],
                                    [['track_name', 0, bytes('Lars Ulrich Challenge', 'utf-8')]]])

    detailed_MIDI_stats = TMIDIX.score2stats([score[0], output,
                                                [['track_name', 0, bytes('Project Los Angeles', 'utf-8')]],
                                                [['track_name', 0, bytes('Lars Ulrich Challenge', 'utf-8')]]])
    
    with open('./LUC-Artificial-Intelligence-Composition' + '.mid', 'wb') as midi_file:
        midi_file.write(midi_data)
        midi_file.close()
    
    print('Done! Enjoy! :)')
    
    print(detailed_MIDI_stats)
    
else:    
    TMIDIX.Tegridy_SONG_to_MIDI_Converter(final_song, 
                                          output_file_name='./LUC-Artificial-Intelligence-Composition',
                                          output_signature = 'Project Los Angeles',
                                          track_name = 'Lars Ulrich Challenge',
                                          number_of_ticks_per_quarter=score[0],
                                          list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0])


# # Congrats! You did it! :)
