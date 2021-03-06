# Copyright (c) 2018-2021, RangerUFO
#
# This file is part of cycle_gan.
#
# cycle_gan is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cycle_gan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cycle_gan.  If not, see <https://www.gnu.org/licenses/>.


import os
import time
import random
import argparse
import mxnet as mx
from dataset import load_dataset, get_batches
from pix2pix_gan import ResnetGenerator, PatchDiscriminator, GANInitializer
from image_pool import ImagePool

def train(dataset, start_epoch, max_epochs, lr_d, lr_g, batch_size, lmda_cyc, lmda_idt, pool_size, context):
    mx.random.seed(int(time.time()))

    print("Loading dataset...", flush=True)
    training_set_a = load_dataset(dataset, "trainA")
    training_set_b = load_dataset(dataset, "trainB")

    gen_ab = ResnetGenerator()
    dis_b = PatchDiscriminator()
    gen_ba = ResnetGenerator()
    dis_a = PatchDiscriminator()
    bce_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
    l1_loss = mx.gluon.loss.L1Loss()

    gen_ab_params_file = "model/{}.gen_ab.params".format(dataset)
    dis_b_params_file = "model/{}.dis_b.params".format(dataset)
    gen_ab_state_file = "model/{}.gen_ab.state".format(dataset)
    dis_b_state_file = "model/{}.dis_b.state".format(dataset)
    gen_ba_params_file = "model/{}.gen_ba.params".format(dataset)
    dis_a_params_file = "model/{}.dis_a.params".format(dataset)
    gen_ba_state_file = "model/{}.gen_ba.state".format(dataset)
    dis_a_state_file = "model/{}.dis_a.state".format(dataset)

    if os.path.isfile(gen_ab_params_file):
        gen_ab.load_parameters(gen_ab_params_file, ctx=context)
    else:
        gen_ab.initialize(GANInitializer(), ctx=context)

    if os.path.isfile(dis_b_params_file):
        dis_b.load_parameters(dis_b_params_file, ctx=context)
    else:
        dis_b.initialize(GANInitializer(), ctx=context)

    if os.path.isfile(gen_ba_params_file):
        gen_ba.load_parameters(gen_ba_params_file, ctx=context)
    else:
        gen_ba.initialize(GANInitializer(), ctx=context)

    if os.path.isfile(dis_a_params_file):
        dis_a.load_parameters(dis_a_params_file, ctx=context)
    else:
        dis_a.initialize(GANInitializer(), ctx=context)

    print("Learning rate of discriminator:", lr_d, flush=True)
    print("Learning rate of generator:", lr_g, flush=True)
    trainer_gen_ab = mx.gluon.Trainer(gen_ab.collect_params(), "Nadam", {
        "learning_rate": lr_g,
        "beta1": 0.5
    })
    trainer_dis_b = mx.gluon.Trainer(dis_b.collect_params(), "Nadam", {
        "learning_rate": lr_d,
        "beta1": 0.5
    })
    trainer_gen_ba = mx.gluon.Trainer(gen_ba.collect_params(), "Nadam", {
        "learning_rate": lr_g,
        "beta1": 0.5
    })
    trainer_dis_a = mx.gluon.Trainer(dis_a.collect_params(), "Nadam", {
        "learning_rate": lr_d,
        "beta1": 0.5
    })

    if os.path.isfile(gen_ab_state_file):
        trainer_gen_ab.load_states(gen_ab_state_file)

    if os.path.isfile(dis_b_state_file):
        trainer_dis_b.load_states(dis_b_state_file)

    if os.path.isfile(gen_ba_state_file):
        trainer_gen_ba.load_states(gen_ba_state_file)

    if os.path.isfile(dis_a_state_file):
        trainer_dis_a.load_states(dis_a_state_file)

    fake_a_pool = ImagePool(pool_size)
    fake_b_pool = ImagePool(pool_size)

    print("Training...", flush=True)
    for epoch in range(start_epoch, max_epochs):
        ts = time.time()

        random.shuffle(training_set_a)
        random.shuffle(training_set_b)

        training_dis_a_L = 0.0
        training_dis_b_L = 0.0
        training_gen_L = 0.0
        training_batch = 0

        for real_a, real_b in get_batches(training_set_a, training_set_b, batch_size, ctx=context):
            training_batch += 1
            
            fake_a, _ = gen_ba(real_b)
            fake_b, _ = gen_ab(real_a)

            with mx.autograd.record():
                real_a_y, real_a_cam_y = dis_a(real_a)
                real_a_L = bce_loss(real_a_y, mx.nd.ones_like(real_a_y, ctx=context))
                real_a_cam_L = bce_loss(real_a_cam_y, mx.nd.ones_like(real_a_cam_y, ctx=context))
                fake_a_y, fake_a_cam_y = dis_a(fake_a_pool.query(fake_a))
                fake_a_L = bce_loss(fake_a_y, mx.nd.zeros_like(fake_a_y, ctx=context))
                fake_a_cam_L = bce_loss(fake_a_cam_y, mx.nd.zeros_like(fake_a_cam_y, ctx=context))
                L = real_a_L + real_a_cam_L + fake_a_L + fake_a_cam_L
                L.backward()
            trainer_dis_a.step(batch_size)
            dis_a_L = mx.nd.mean(L).asscalar()
            if dis_a_L != dis_a_L:
                raise ValueError()

            with mx.autograd.record():
                real_b_y, real_b_cam_y = dis_b(real_b)
                real_b_L = bce_loss(real_b_y, mx.nd.ones_like(real_b_y, ctx=context))
                real_b_cam_L = bce_loss(real_b_cam_y, mx.nd.ones_like(real_b_cam_y, ctx=context))
                fake_b_y, fake_b_cam_y = dis_b(fake_b_pool.query(fake_b))
                fake_b_L = bce_loss(fake_b_y, mx.nd.zeros_like(fake_b_y, ctx=context))
                fake_b_cam_L = bce_loss(fake_b_cam_y, mx.nd.zeros_like(fake_b_cam_y, ctx=context))
                L = real_b_L + real_b_cam_L + fake_b_L + fake_b_cam_L
                L.backward()
            trainer_dis_b.step(batch_size)
            dis_b_L = mx.nd.mean(L).asscalar()
            if dis_b_L != dis_b_L:
                raise ValueError()

            with mx.autograd.record():
                fake_a, gen_a_cam_y = gen_ba(real_b)
                fake_a_y, fake_a_cam_y = dis_a(fake_a)
                gan_a_L = bce_loss(fake_a_y, mx.nd.ones_like(fake_a_y, ctx=context))
                gan_a_cam_L = bce_loss(fake_a_cam_y, mx.nd.ones_like(fake_a_cam_y, ctx=context))
                rec_b, _ = gen_ab(fake_a)
                cyc_b_L = l1_loss(rec_b, real_b)
                idt_a, idt_a_cam_y = gen_ba(real_a)
                idt_a_L = l1_loss(idt_a, real_a)
                gen_a_cam_L = bce_loss(gen_a_cam_y, mx.nd.ones_like(gen_a_cam_y, ctx=context)) + bce_loss(idt_a_cam_y, mx.nd.zeros_like(idt_a_cam_y, ctx=context))
                gen_ba_L = gan_a_L + gan_a_cam_L + cyc_b_L * lmda_cyc + idt_a_L * lmda_cyc * lmda_idt + gen_a_cam_L
                fake_b, gen_b_cam_y = gen_ab(real_a)
                fake_b_y, fake_b_cam_y = dis_b(fake_b)
                gan_b_L = bce_loss(fake_b_y, mx.nd.ones_like(fake_b_y, ctx=context))
                gan_b_cam_L = bce_loss(fake_b_cam_y, mx.nd.ones_like(fake_b_cam_y, ctx=context))
                rec_a, _ = gen_ba(fake_b)
                cyc_a_L = l1_loss(rec_a, real_a)
                idt_b, idt_b_cam_y = gen_ab(real_b)
                idt_b_L = l1_loss(idt_b, real_b)
                gen_b_cam_L = bce_loss(gen_b_cam_y, mx.nd.ones_like(gen_b_cam_y, ctx=context)) + bce_loss(idt_b_cam_y, mx.nd.zeros_like(idt_b_cam_y, ctx=context))
                gen_ab_L = gan_b_L + gan_b_cam_L + cyc_a_L * lmda_cyc + idt_b_L * lmda_cyc * lmda_idt + gen_b_cam_L
                L = gen_ba_L + gen_ab_L
                L.backward()
            trainer_gen_ba.step(batch_size)
            trainer_gen_ab.step(batch_size)
            gen_L = mx.nd.mean(L).asscalar()
            if gen_L != gen_L:
                raise ValueError()

            training_dis_a_L += dis_a_L
            training_dis_b_L += dis_b_L
            training_gen_L += gen_L
            print("[Epoch %d  Batch %d]  dis_a_loss %.10f  dis_b_loss %.10f  gen_loss %.10f  elapsed %.2fs" % (
                epoch, training_batch, dis_a_L, dis_b_L, gen_L, time.time() - ts
            ), flush=True)

        print("[Epoch %d]  training_dis_a_loss %.10f  training_dis_b_loss %.10f  training_gen_loss %.10f  duration %.2fs" % (
            epoch + 1, training_dis_a_L / training_batch, training_dis_b_L / training_batch, training_gen_L / training_batch, time.time() - ts
        ), flush=True)

        gen_ab.save_parameters(gen_ab_params_file)
        gen_ba.save_parameters(gen_ba_params_file)
        dis_a.save_parameters(dis_a_params_file)
        dis_b.save_parameters(dis_b_params_file)
        trainer_gen_ab.save_states(gen_ab_state_file)
        trainer_gen_ba.save_states(gen_ba_state_file)
        trainer_dis_a.save_states(dis_a_state_file)
        trainer_dis_b.save_states(dis_b_state_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a cycle_gan trainer.")
    parser.add_argument("--dataset", help="set the dataset used by the trainer (default: vangogh2photo)", type=str, default="vangogh2photo")
    parser.add_argument("--start_epoch", help="set the start epoch (default: 0)", type=int, default=0)
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--lr_d", help="set the learning rate of discriminator (default: 0.0003)", type=float, default=0.0003)
    parser.add_argument("--lr_g", help="set the learning rate of generator (default: 0.0001)", type=float, default=0.0001)
    parser.add_argument("--batch_size", help="set the batch size (default: 32)", type=int, default=32)
    parser.add_argument("--lmda_cyc", help="set the lambda of cycle loss (default: 10.0)", type=float, default=10.0)
    parser.add_argument("--lmda_idt", help="set the lambda of identity loss (default: 0.5)", type=float, default=0.5)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    while True:
        try:
            train(
                dataset = args.dataset,
                start_epoch = args.start_epoch,
                max_epochs = args.max_epochs,
                lr_d = args.lr_d,
                lr_g = args.lr_g,
                batch_size = args.batch_size,
                lmda_cyc = args.lmda_cyc,
                lmda_idt = args.lmda_idt,
                pool_size = 50,
                context = context
            )
            break;
        except ValueError:
            print("Oops! The value of loss become NaN...")
