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

# In[ ]:


#@title Import all needed modules

print('Loading needed modules. Please wait...')
import os
from datetime import datetime
import secrets
import copy
import tqdm
from tqdm import auto

print('Loading TMIDIX module...')
import TMIDIX
print('Loading GPT2RGA module...')
from GPT2RGA import *


# # Algorithmic Drums Matching

# In[ ]:


#print('Loading MIDI file...')

source_MIDI_file = 'Nothing Else Matters.mid'
number_of_drums_per_chord = 3 # drums pitches per chord/note

mel_crd_f = []

score = TMIDIX.midi2ms_score(open(source_MIDI_file, 'rb').read())

events_matrix = []
itrack = 1

while itrack < len(score):
    for event in score[itrack]:
        if event[0] == 'note' and event[3] != 9:
            events_matrix.append(event)
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
        for d in drums[idx][:number_of_drums_per_chord]:
            # if d < 40:
            drum = [['note', mel_crd_f[i][0][1], mel_crd_f[i][0][2], 9, d, 100 ]]
            final_song.extend(drum)
            drums_map.extend(drum)
        idx = -1
        
TMIDIX.Tegridy_SONG_to_MIDI_Converter(final_song, 
                                      output_file_name='./LUC-Algorithmic-Composition',
                                      output_signature = 'Project Los Angeles', 
                                      track_name = 'Lars Ulrich Challenge',
                                      number_of_ticks_per_quarter=500,
                                      list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0])


# # Artificial Intelligence Drums Matching

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


#print('Loading MIDI file...')

source_MIDI_file = 'Nothing Else Matters.mid' # soutce MIDI file
number_of_drums_per_chord = -1 # drums pitches per chord/note
memory_tokens = 64 # recommended 32-64



mel_crd_f = []
score = TMIDIX.midi2ms_score(open(source_MIDI_file, 'rb').read())

events_matrix = []
itrack = 1

while itrack < len(score):
    for event in score[itrack]:
        if event[0] == 'note' and event[3] != 9: # reading all notes events except for the drums
            events_matrix.append(event)
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

for i in tqdm(range(len(src_pitches))):
#for i in tqdm([0]):
    
    
    final_song.extend(mel_crd_f[i])
    if memory_tokens > 0:
        number_of_tokens_to_generate = memory_tokens+32 #@param {type:"slider", min:8, max:1024, step:8}
        data = out[-memory_tokens:] + [388] + src_pitches[i] + [128]
    else:
        number_of_tokens_to_generate = 32 #@param {type:"slider", min:8, max:1024, step:8}
        data = [388] + src_pitches[i] + [128]
        
    rand_seq = model.generate(torch.Tensor(data), target_seq_length=number_of_tokens_to_generate, stop_token=128)
    out = rand_seq[0].cpu().numpy().tolist()
    out_cc = out[out.index(128, len(data)-1)+1:]

    for d in out_cc[:number_of_drums_per_chord]:
        drum = [['note', mel_crd_f[i][0][1], mel_crd_f[i][0][2], 9, d, 100 ]]
        final_song.extend(drum)
        
TMIDIX.Tegridy_SONG_to_MIDI_Converter(final_song, 
                                      output_file_name='./LUC-Artificial-Intelligence-Composition',
                                      output_signature = 'Project Los Angeles',
                                      track_name = 'Lars Ulrich Challenge',
                                      number_of_ticks_per_quarter=500,
                                      list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0])

