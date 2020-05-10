# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import random
import io
import exiv2
import pickle
import time
import glob
import os
import itertools

import pretrained_networks




#----------------------------------------------------------------------------
def read_metadata(path):
    image = exiv2.ImageFactory.open(path)
    image.readMetadata()
    ipc_data = image.iptcData()
    data = ipc_data[b"Iptc.Application2.Caption"]
    x = pickle.loads(data.toString())
    return x
    
def save_with_metadata(im, filename, metadata, timestamp = True):
    bio = io.BytesIO() # this is a file object
    if -1 == filename.find("png"):
        im.save(bio, format="jpeg", quality=85)
    else:
        im.save(bio, format="png")
    imdata = bio.getvalue()

    bio2 = io.BytesIO() # this is a file object
    pickle.dump(metadata, bio2)
    pickle_data = bio2.getvalue()
     
    im = exiv2.ImageFactory.open(imdata)
    im.readMetadata()
    new_iptc_data = exiv2.IptcData()
    new_iptc_data[b"Iptc.Application2.Caption"] = pickle_data
    im.setIptcData(new_iptc_data)
    im.writeMetadata()

    imio = im.io()
    size = imio.size()
    buffer = imio.read(size)
    if timestamp    :
        filename = str(int(time.time()*100)) + '_' + filename
    path = dnnlib.make_run_dir_path(filename)    
    print(path)
    with open(path, "wb") as f:
        f.write(buffer)
        

def regenerate_folder(network_pkl, input_dir):
    files = glob.glob(input_dir + "/*.png") + glob.glob(input_dir + "/*.jpg")
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    
    for f in files:
        metadata = read_metadata(f)
        print(metadata.keys())
        z = metadata['z']
        truncation_psi = metadata['truncation_psi']
        if 'noise_vars' in metadata:
            noise_vars = metadata['noise_vars']
            vars = {var:noise_vars[name] for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')}
        else:
            rnd = np.random.RandomState(metadata['noise_seed'])
            vars = {var: rnd.randn(*var.shape.as_list()) for var in noise_vars}        
        Gs_kwargs.truncation_psi = truncation_psi
        tflib.set_vars(vars) # [height, width]           
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]            
        im = PIL.Image.fromarray(images[0], 'RGB')  #.save(dnnlib.make_run_dir_path('varysingle_%04d.png' % i))
        fn = os.path.splitext(os.path.basename(f))[0] + '-b.png'
        save_with_metadata(im, fn, metadata, False)
            
def vary_folder(network_pkl, input_dir, q, count, psi0, psi1, repeat):
    files = glob.glob(input_dir + "/*.png") + glob.glob(input_dir + "/*.jpg")
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    
    random.shuffle(files)
    for cycle in range(repeat):
        for f in files:
            metadata = read_metadata(f)
            z = metadata['z']
            truncation_psi = metadata['truncation_psi']
            noise_seed = metadata['noise_seed']
            rnd = np.random.RandomState(noise_seed)
            vars = {var: rnd.randn(*var.shape.as_list()) for var in noise_vars}        
            tflib.set_vars(vars) # [height, width]    
            '''        
            truncation_psi2 = truncation_psi + np.random.randn()/q/10
            if truncation_psi2 < 0.3:
                truncation_psi2 = 0.3
            elif truncation_psi2 > 1:g/
                truncation_psi2 = 1        
            '''
            truncation_psi2 = psi0 + (psi1-psi0)*np.random.random()        
            Gs_kwargs.truncation_psi = truncation_psi2                   
            
            for i in range(count):
                z2 = (z + np.random.randn(*z.shape)/q) / (1 + 1/(q*q))       
                print(z2.shape)
                images = Gs.run(z2, None, **Gs_kwargs) # [minibatch, height, width, channel]            
                #g = tf.get_default_graph()
                #g.get_tensor_by_name('Gs/_Run/Gs/G_mapping/dlatents_out:0')

                im = PIL.Image.fromarray(images[0], 'RGB')  #.save(dnnlib.make_run_dir_path('varysingle_%04d.png' % i))
                fn = '-vf.jpg'
                metadata = {'z':z2, 'truncation_psi':truncation_psi2, 'noise_seed':noise_seed}            
                save_with_metadata(im, fn, metadata, True)
            
            
def vary_seeds(network_pkl, seeds, psi0, psi1, save_noise, q, count):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    #noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    noise_vars_dict = {name:var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')}

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    #if truncation_psi is not None:
    #    Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        z2 = z
        noise_seed = rnd.randint(0,1000000)
        rnd_noise = np.random.RandomState(noise_seed)
        vars = {var: rnd_noise.randn(*var.shape.as_list()) for name, var in noise_vars_dict.items()}
        truncation_psi = psi0 + (psi1-psi0)*np.random.random()
        Gs_kwargs.truncation_psi = truncation_psi        
        for i in range (count):
            tflib.set_vars(vars) # [height, width]           
            images = Gs.run(z2, None, **Gs_kwargs) # [minibatch, height, width, channel]            
            im = PIL.Image.fromarray(images[0], 'RGB')  #.save(dnnlib.make_run_dir_path('varysingle_%04d.png' % i))
            fn = 'vary_seed_%08d.jpg'%seed
            metadata = {'z':z2, 'truncation_psi':truncation_psi}            
            if save_noise:
                metadata_noise_vars = { name : vars[var] for name, var in noise_vars_dict.items()}
                metadata['noise_vars'] = metadata_noise_vars
            else:
                metadata['noise_seed'] = noise_seed
            save_with_metadata(im, fn, metadata, True)            
            z2 = (z + np.random.randn(*z.shape)/q) / (1 + 1/(q*q))            
          
def interpolate_seeds(network_pkl, seeds, psi0, psi1, count, first_against_all):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    L = []
    for seed in seeds:
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]        
        L.append((seed, z))
        #noise_seed = rnd.randint(0,1000000)
        #rnd_noise = np.random.RandomState(noise_seed)
        #vars = {var: rnd_noise.randn(*var.shape.as_list()) for var in noise_vars}        
        #L.append((seed, z, noise_seed, vars))
        
    if first_against_all:
        it = itertools.product([L[0]],L[1:])
    else:
        it = itertools.combinations(L, 2)
        
    for x, y in it:  
        truncation_psi = psi0 + (psi1-psi0)*np.random.random()
        Gs_kwargs.truncation_psi = truncation_psi        
        for i in range(count):
            noise_seed = rnd.randint(0,1000000)
            rnd_noise = np.random.RandomState(noise_seed)
            vars = {var: rnd_noise.randn(*var.shape.as_list()) for var in noise_vars}        
        
            r = (i+1)/(count+1)
            #r = random.random()
            z = (1-r)*x[1] + r*y[1]
            tflib.set_vars(vars) # [height, width]           
            images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            im = PIL.Image.fromarray(images[0], 'RGB')  #.save(dnnlib.make_run_dir_path('varysingle_%04d.png' % i))            
            fn = '{:010d}-{:010d}-{:03d}.png'.format(x[0], y[0], int(r*1000))
            metadata = {'z':z, 'truncation_psi':truncation_psi, 'noise_seed':noise_seed}            
            save_with_metadata(im, fn, metadata, True)               

def interpolate_folder(network_pkl, input_dir, input_dir2, count):
    files = glob.glob(input_dir + "/*.png") + glob.glob(input_dir + "/*.jpg")
    L = []
    for f in files:
        metadata = read_metadata(f)
        z = metadata['z']
        truncation_psi = metadata['truncation_psi']
        L.append((z, truncation_psi))
    if input_dir2:
        files2 = glob.glob(input_dir2 + "/*.png") + glob.glob(input_dir2 + "/*.jpg")
        L2 = []
        for f in files2:
            metadata = read_metadata(f)
            z = metadata['z']
            truncation_psi = metadata['truncation_psi']
            L2.append((z, truncation_psi))    
    else:
        L2 = []
    
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    
        
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    random.shuffle(L)
    if L2:
        random.shuffle(L2)
        it = itertools.product(L, L2)    
    else:
        it = itertools.combinations(L, 2)
        
    for x, y in it:  
        psi0, psi1 = x[1], y[1]
        truncation_psi = psi0 + (psi1-psi0)*np.random.random()
        Gs_kwargs.truncation_psi = truncation_psi        
        noise_seed = np.random.randint(0,1000000)
        rnd_noise = np.random.RandomState(noise_seed)
        vars = {var: rnd_noise.randn(*var.shape.as_list()) for var in noise_vars}        
        tflib.set_vars(vars) # [height, width]                   
        for i in range(count):        
            r = (i+1)/(count+1)
            #r = random.random()
            z = (1-r)*x[0] + r*y[0]
            images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            im = PIL.Image.fromarray(images[0], 'RGB')  #.save(dnnlib.make_run_dir_path('varysingle_%04d.png' % i))            
            fn = 'if_{:03d}.jpg'.format(int(r*1000))
            metadata = {'z':z, 'truncation_psi':truncation_psi, 'noise_seed':noise_seed}            
            save_with_metadata(im, fn, metadata, True)               
        
def generate_from_npy(network_pkl, input_dir, truncation_psi):
    import generator_model
    import tensorflow as tf
    
    files = glob.glob(input_dir + "/*.npy")
    L = []
    for f in files:
        data = np.load(f) 
        assert data.shape == (18, 512)
        L.append(data)
        
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    '''    
    for x in tf.global_variables():
        if 'latent' in x.name: 
            print(x)
    dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
    '''
    
    
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi
    
    noise_seed = np.random.randint(0,100000)
    noise_rnd = np.random.RandomState(noise_seed)
    vars = {var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}
    tflib.set_vars(vars) # [height, width]           
    
    GM = generator_model.Generator(Gs, 1)
    
    for data in L:
        z = data
        #images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        images = GM.generate_images(z)
        im = PIL.Image.fromarray(images[0], 'RGB')  #.save(dnnlib.make_run_dir_path('varysingle_%04d.png' % i))            
        fn = 'npy.jpg'
        metadata = {'z':z, 'truncation_psi':truncation_psi, 'noise_seed':noise_seed}            
        save_with_metadata(im, fn, metadata, True)               
        
def generate_images(network_pkl, seeds, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    L = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]        
        vars = {var: rnd.randn(*var.shape.as_list()) for var in noise_vars}
        L.append((seed, z, vars))
    L0 = L[0]
    L = L[1:]
        
    for i in range (200):
        random.shuffle(L)
        #L0, L1 = L[0], L[1]
        L1 = L[0]
        r = random.random()
        z = r*L0[1] + (1-r)*L1[1]
        vars = { var : r*L0[2][var] + (1-r)*L1[2][var] for var in noise_vars }
        tflib.set_vars(vars) # [height, width]           
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        path = 'seed{:010d}-{:010d}-{:03d}.png'.format(L0[0], L1[0], int(r*1000))
        print(path)
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path(path))
            
def generate_images(network_pkl, seeds, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))
        
#----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d.png' % (row_seed, col_seed)))

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(dnnlib.make_run_dir_path('grid.png'))

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=66,230,389,1518 --truncation-psi=1.0

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=6000-6025 --truncation-psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    default_pkl = r"F:\DEEP\styleganV2\stylegan2-ffhq-config-f.pkl"
    
    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', default=default_pkl)
    parser_generate_images.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_generate_from_npy = subparsers.add_parser('generate-from-npy', help='Generate images')
    parser_generate_from_npy.add_argument('--network', help='Network pickle filename', dest='network_pkl', default=default_pkl)
    parser_generate_from_npy.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_from_npy.add_argument('--input-dir', help='Input folder', required=True)
    parser_generate_from_npy.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')


    parser_regenerate_folder = subparsers.add_parser('regenerate-folder', help='Generate images')
    parser_regenerate_folder.add_argument('--network', help='Network pickle filename', dest='network_pkl', default=default_pkl)
    parser_regenerate_folder.add_argument('--input-dir', help='Input folder', required=True)
    parser_regenerate_folder.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    
    parser_vary_seeds = subparsers.add_parser('vary-seeds', help='Generate images')
    parser_vary_seeds.add_argument('--network', help='Network pickle filename', dest='network_pkl', default=default_pkl)
    parser_vary_seeds.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_vary_seeds.add_argument('--psi0', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_vary_seeds.add_argument('--psi1', type=float, help='Truncation psi (default: %(default)s)', default=0.55)
    parser_vary_seeds.add_argument('--count', type=int, default=20)
    parser_vary_seeds.add_argument('--q', type=float, help='', default=5)
    parser_vary_seeds.add_argument('--save-noise', default=False)
    parser_vary_seeds.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_vary_folder = subparsers.add_parser('vary-folder', help='Generate images')
    parser_vary_folder.add_argument('--network', help='Network pickle filename', dest='network_pkl', default=default_pkl)
    parser_vary_folder.add_argument('--input-dir', help='Input folder', required=True)    
    parser_vary_folder.add_argument('--count', type=int, default=20)
    parser_vary_folder.add_argument('--repeat', type=int, default=1)
    parser_vary_folder.add_argument('--q', type=float, help='', default=5)
    parser_vary_folder.add_argument('--psi0', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_vary_folder.add_argument('--psi1', type=float, help='Truncation psi (default: %(default)s)', default=0.6)    
    parser_vary_folder.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    
    parser_interpolate_folder = subparsers.add_parser('interpolate-folder', help='Generate images')
    parser_interpolate_folder.add_argument('--network', help='Network pickle filename', dest='network_pkl', default=default_pkl)
    parser_interpolate_folder.add_argument('--input-dir', help='Input folder', required=True)    
    parser_interpolate_folder.add_argument('--input-dir2', help='Input folder', default=None)    
    parser_interpolate_folder.add_argument('--count', type=int, default=20)
    parser_interpolate_folder.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    
    parser_interpolate_seeds = subparsers.add_parser('interpolate-seeds', help='Generate images')
    parser_interpolate_seeds.add_argument('--network', help='Network pickle filename', dest='network_pkl', default=default_pkl)
    parser_interpolate_seeds.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_interpolate_seeds.add_argument('--psi0', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_interpolate_seeds.add_argument('--psi1', type=float, help='Truncation psi (default: %(default)s)', default=0.55)
    parser_interpolate_seeds.add_argument('--count', type=int, default=20)
    parser_interpolate_seeds.add_argument('--first-against-all', type=int, default=0)
    parser_interpolate_seeds.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')


    parser_style_mixing_example = subparsers.add_parser('style-mixing-example', help='Generate style mixing video')
    parser_style_mixing_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', default=default_pkl)
    parser_style_mixing_example.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_example.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_style_mixing_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'generate-images': 'run_generator.generate_images',
        'generate-from-npy': 'run_generator.generate_from_npy',
        'vary-seeds': 'run_generator.vary_seeds',
        'vary-folder': 'run_generator.vary_folder',        
        'interpolate-seeds': 'run_generator.interpolate_seeds',
        'interpolate-folder': 'run_generator.interpolate_folder',
        'regenerate-folder': 'run_generator.regenerate_folder',
        'style-mixing-example': 'run_generator.style_mixing_example'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
