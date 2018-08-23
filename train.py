import os
import time
import argparse
import mxnet as mx
from dataset import load_dataset, get_batches
from pix2pix_gan import UnetGenerator, Discriminator, WassersteinLoss

def train(dataset, max_epochs, learning_rate, batch_size, filters, lmda_cyc, lmda_idt, context):
    mx.random.seed(int(time.time()))

    print("Loading dataset...", flush=True)
    training_set_a = load_dataset(dataset, "trainA")
    training_set_b = load_dataset(dataset, "trainB")

    gen_ab = UnetGenerator(3, filters)
    dis_b = Discriminator(filters)
    gen_ba = UnetGenerator(3, filters)
    dis_a = Discriminator(filters)
    wgan_loss = WassersteinLoss()
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
        gen_ab.initialize(mx.init.Xavier(), ctx=context)

    if os.path.isfile(dis_b_params_file):
        dis_b.load_parameters(dis_b_params_file, ctx=context)
    else:
        dis_b.initialize(mx.init.Xavier(), ctx=context)

    if os.path.isfile(gen_ba_params_file):
        gen_ba.load_parameters(gen_ba_params_file, ctx=context)
    else:
        gen_ba.initialize(mx.init.Xavier(), ctx=context)

    if os.path.isfile(dis_a_params_file):
        dis_a.load_parameters(dis_a_params_file, ctx=context)
    else:
        dis_a.initialize(mx.init.Xavier(), ctx=context)

    print("Learning rate:", learning_rate, flush=True)
    trainer_gen_ab = mx.gluon.Trainer(gen_ab.collect_params(), "RMSProp", {
        "learning_rate": learning_rate
    })
    trainer_dis_b = mx.gluon.Trainer(dis_b.collect_params(), "RMSProp", {
        "learning_rate": learning_rate,
        "clip_weights": 0.01
    })
    trainer_gen_ba = mx.gluon.Trainer(gen_ba.collect_params(), "RMSProp", {
        "learning_rate": learning_rate
    })
    trainer_dis_a = mx.gluon.Trainer(dis_a.collect_params(), "RMSProp", {
        "learning_rate": learning_rate,
        "clip_weights": 0.01
    })

    if os.path.isfile(gen_ab_state_file):
        trainer_gen_ab.load_states(gen_ab_state_file)

    if os.path.isfile(dis_b_state_file):
        trainer_dis_b.load_states(dis_b_state_file)

    if os.path.isfile(gen_ba_state_file):
        trainer_gen_ba.load_states(gen_ba_state_file)

    if os.path.isfile(dis_a_state_file):
        trainer_dis_a.load_states(dis_a_state_file)

    print("Training...", flush=True)
    for epoch in range(max_epochs):
        ts = time.time()

        training_dis_L = 0.0
        training_gen_L = 0.0
        training_batch = 0

        for batch_a, batch_b in get_batches(training_set_a, training_set_b, batch_size):
            training_batch += 1
            
            real_a = batch_a.as_in_context(context)
            real_b = batch_b.as_in_context(context)
            fake_a = gen_ba(real_b)
            fake_b = gen_ab(real_a)

            with mx.autograd.record():
                real_a_y = dis_a(real_a)
                fake_a_y = dis_a(fake_a)
                L = wgan_loss(fake_a_y, real_a_y)
                L.backward()
            trainer_dis_a.step(batch_size)
            dis_a_L = mx.nd.mean(L).asscalar()
            if dis_a_L != dis_a_L:
                raise ValueError()

            with mx.autograd.record():
                real_b_y = dis_b(real_b)
                fake_b_y = dis_b(fake_b)
                L = wgan_loss(fake_b_y, real_b_y)
                L.backward()
            trainer_dis_b.step(batch_size)
            dis_b_L = mx.nd.mean(L).asscalar()
            if dis_b_L != dis_b_L:
                raise ValueError()

            with mx.autograd.record():
                fake_a = gen_ba(real_b)
                fake_a_y = dis_a(fake_a)
                rec_b = gen_ab(fake_a)
                cyc_b_L = l1_loss(rec_b, real_b)
                idt_a = gen_ba(real_a)
                idt_a_L = l1_loss(idt_a, real_a)
                L = wgan_loss(fake_a_y) + cyc_b_L * lmda_cyc + idt_a_L * lmda_cyc * lmda_idt
                L.backward()
            trainer_gen_ba.step(batch_size)
            gen_ba_L = mx.nd.mean(L).asscalar()
            if gen_ba_L != gen_ba_L:
                raise ValueError()

            with mx.autograd.record():
                fake_b = gen_ab(real_a)
                fake_b_y = dis_b(fake_b)
                rec_a = gen_ba(fake_b)
                cyc_a_L = l1_loss(rec_a, real_a)
                idt_b = gen_ab(real_b)
                idt_b_L = l1_loss(idt_b, real_b)
                L = wgan_loss(fake_b_y) + cyc_a_L * lmda_cyc + idt_b_L * lmda_cyc * lmda_idt
                L.backward()
            trainer_gen_ab.step(batch_size)
            gen_ab_L = mx.nd.mean(L).asscalar()
            if gen_ab_L != gen_ab_L:
                raise ValueError()

            dis_L = dis_a_L + dis_b_L
            gen_L = gen_ba_L + gen_ab_L
            training_dis_L += dis_L
            training_gen_L += gen_L
            print("[Epoch %d  Batch %d]  dis_loss %.10f  gen_loss %.10f  elapsed %.2fs" % (
                epoch, training_batch, dis_L, gen_L, time.time() - ts
            ), flush=True)

        print("[Epoch %d]  training_dis_loss %.10f  training_gen_loss %.10f  duration %.2fs" % (
            epoch + 1, training_dis_L / training_batch, training_gen_L / training_batch, time.time() - ts
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
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--learning_rate", help="set the learning rate (default: 0.00005)", type=float, default=0.00005)
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
                max_epochs = args.max_epochs,
                learning_rate = args.learning_rate,
                batch_size = 1,
                filters = 64,
                lmda_cyc = 10,
                lmda_idt = 0.5,
                context = context
            )
            break;
        except ValueError:
            print("Oops! The value of loss become NaN...")
