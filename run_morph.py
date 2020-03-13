# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import tensorflow as tf
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import pretrained_networks

from training import misc

#----------------------------------------------------------------------------

def generate_morph(network_pkl, start, end, num, prefix):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    start_data = np.load(start)
    end_data = np.load(end)

    # Build image graph
    dlatents_var = tf.Variable(tf.zeros(start_data['arr_0'].shape), name='dlatents_var')
    images_expr = Gs.components.synthesis.get_output_for(dlatents_var, randomize_noise=False)

    for i in range(num):
        p = i*1.0/(num-1)

        dlatent = start_data['arr_0'] + p*(end_data['arr_0']-start_data['arr_0'])
        # noise = [ns + p*(ne-ns) for ns, ne in zip(start_data[1:], end_data[1:])]

        print('Generating image %d/%d ...' % (i+1, num))
        images = tflib.run(images_expr, {dlatents_var: dlatent})
        misc.save_image_grid(images, 'morph/%s_%03d.png' % (prefix, i+1), drange=[-1,1])

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 morph generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_morph = subparsers.add_parser('generate-morph', help='Generate morph images')
    parser_generate_morph.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_morph.add_argument('--start', help='Start projection', required=True)
    parser_generate_morph.add_argument('--end', help='End projection', required=True)
    parser_generate_morph.add_argument('--num', type=int, help='Number of images in output', required=True)
    parser_generate_morph.add_argument('--prefix', help='Morph file name prefix (default: %(default)s)', default='morph')

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
    sc.run_desc = subcmd

    func_name_map = {
        'generate-morph': 'run_morph.generate_morph'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
