import numpy as np
import time as tm

from cocoa.lib import logstats
from cocoa.lib.bleu import compute_bleu
from neural_model.batcher import DialogueBatcher
# from cocoa.model.learner import Learner as BaseLearner, add_learner_arguments

from torch.nn import NLLLoss, parameter
from torch import optim

def add_learner_arguments(parser):
    parser.add_argument('--optimizer', default='sgd', help='Optimization method')
    parser.add_argument('--sample-targets', action='store_true', help='Sample targets from candidates')
    parser.add_argument('--grad-clip', type=int, default=5, help='Min and max values of gradients')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--min-epochs', type=int, default=10, help='Number of training epochs to run before checking for early stop')
    parser.add_argument('--max-epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--num-per-epoch', type=int, default=None, help='Number of examples per epoch')
    parser.add_argument('--print-every', type=int, default=1, help='Number of examples between printing training loss')
    parser.add_argument('--init-from', default="data/first-trial", help='Initial parameters')
    parser.add_argument('--checkpoint', default='.', help='Directory to save learned models')
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU or not')
    parser.add_argument('--summary-dir', default='/tmp', help='Path to summary logs')
    parser.add_argument('--eval-modes', nargs='*', default=('loss',), help='What to evaluate {loss, generation}')

class Learner(object):
    def __init__(self, args, encoder, decoder, vocab, use_cuda):
        self.encoder = encoder.cuda() if use_cuda else encoder
        self.decoder = decoder.cuda() if use_cuda else decoder
        self.enc_optimizer = self.add_optimizers(self.encoder, args)
        self.dec_optimizer = self.add_optimizers(self.decoder, args)

        self.vocab = vocab
        self.summary_dir = args.summary_dir
        self.verbose = args.verbose
        # self.evaluator = evaluator
        self.create_embeddings()

        # self.train_data = DialogueBatcher(vocab, "train")
        # self.val_data = DialogueBatcher(vocab, "valid")
        self.toy_data = DialogueBatcher(vocab, "toy")
        # self.test_data = DialogueBatcher(vocab, "test")

        self.use_cuda = use_cuda
        self.criterion = NLLLoss()
        self.teach_ratio = args.teacher_forcing_ratio

        self.train_iterations = self.toy_data.num_per_epoch * args.min_epochs
        self.val_iterations = self.toy_data.num_per_epoch * args.min_epochs
        self.train_print_every = 500
        self.val_print_every = 700

    def _run_batch(self, dialogue_batch, sess, summary_map, test=True):
        raise NotImplementedError

    def test_loss(self, sess, test_data, num_batches):
        '''
        Return the cross-entropy loss.
        '''
        summary_map = {}
        for i in xrange(num_batches):
            dialogue_batch = test_data.next()
            self._run_batch(dialogue_batch, sess, summary_map, test=True)
        return summary_map

    def add_optimizers(self, model, args):
        optimizers = {'adagrad': optim.Adagrad,
                          'sgd': optim.SGD,
                         'adam': optim.Adam}
        optimizer = optimizers[args.optimizer]
        return optimizer(model.parameters(), args.learning_rate)

    def create_embeddings(self):
        # embedding is always tied, can change this to decouple in the future
        vocab_matrix = self.encoder.create_embedding(self.vocab.size)
        self.decoder.create_embedding(vocab_matrix, self.vocab.size)

    def _print_batch(self, batch, preds, loss):
        batcher = self.data.dialogue_batcher
        textint_map = self.data.textint_map
        # Go over each example in the batch
        print '-------------- Batch ----------------'
        for i in xrange(batch['size']):
            success = batcher.print_batch(batch, i, textint_map, preds)
        print 'BATCH LOSS:', loss

    def eval(self, sess, name, test_data, num_batches, output=None, modes=('loss',)):
        print '================== Eval %s ==================' % name
        results = {}

        if 'loss' in modes:
            print '================== Loss =================='
            start_time = time.time()
            summary_map = self.test_loss(sess, test_data, num_batches)
            results = self.collect_summary_test(summary_map, results)
            results_str = ' '.join(['{}={:.4f}'.format(k, v) for k, v in results.iteritems()])
            print '%s time(s)=%.4f' % (results_str, time.time() - start_time)

        if 'generation' in modes:
            print '================== Generation =================='
            start_time = time.time()
            res = self.evaluator.test_response_generation(sess, test_data, num_batches, output=output)
            results.update(res)
            # TODO: hacky. for LM only.
            if len(results) > 0:
                print '%s time(s)=%.4f' % (self.evaluator.stats2str(results), time.time() - start_time)
        return results

    def learn(self, args):
        start = tm.time()
        assert args.min_epochs <= args.max_epochs

        # Gradient
        save_model = False
        train_steps, train_losses = [], []
        val_steps, val_losses = [], []
        bleu_scores, accuracy = [], []

        iters = self.train_iterations
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        enc_scheduler = StepLR(self.enc_optimizer, step_size=iters/3, gamma=0.2)
        dec_scheduler = StepLR(self.dec_optimizer, step_size=iters/3, gamma=0.2)

        for iter in range(1, iters + 1):
            enc_scheduler.step()
            dec_scheduler.step()

            training_pair = self.train_data.get_batch()
            input_variable = training_pair[0]
            output_variable = training_pair[1]

            starting_checkpoint(iter)
            loss = train(input_variable, output_variable)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%d%% complete %s, Train Loss: %.4f' % ((iter / iters * 100),
                    timeSince(start, iter / iters), print_loss_avg))
                train_losses.append(print_loss_avg)
                train_steps.append(iter)

            if iter % val_every == 0:
                val_steps.append(iter)
                batch_val_loss, batch_bleu, batch_success = [], [], []
                for iter in range(1, self.val_iterations + 1):
                    val_pair = validation_pairs[iter - 1]
                    val_input = val_pair[0]
                    val_output = val_pair[1]
                    val_loss, bleu_score, turn_success = validate(val_input, \
                          val_output, task)
                    batch_val_loss.append(val_loss)
                    batch_bleu.append(bleu_score)
                    batch_success.append(turn_success)

                avg_val_loss, avg_bleu, avg_success = evaluate.batch_processing(
                                              batch_val_loss, bleu_score, batch_success)
                # val_losses.append(avg_val_loss)
                bleu_scores.append(avg_bleu)
                accuracy.append(avg_success)
                # save_path = os.path.join(args.checkpoint, 'tf_model.ckpt')
                # best_saver = tf.train.Saver(max_to_keep=1)
                # best_checkpoint = args.checkpoint+'-best'

        time_past(start)
        return train_steps, train_losses, val_steps, val_losses, accuracy

    def run_inference(sources, targets, teach_ratio):
        loss = 0
        encoder_hidden = self.encoder.initHidden()
        encoder_length = sources.size()[0]
        encoder_outputs, encoder_hidden = self.encoder(sources, encoder_hidden)

        decoder_hidden = encoder_hidden
        decoder_length = targets.size()[0]
        decoder_input = smart_variable(torch.LongTensor([[vocab.SOS_token]]))
        decoder_context = smart_variable(torch.zeros(1, 1, self.decoder.hidden_size))
        # visual = torch.zeros(encoder_length, decoder_length)
        predictions = []

        for di in range(decoder_length):
            use_teacher_forcing = random.random() < self.teach_ratio
            decoder_output, decoder_context, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs)

            # visual[:, di] = attn_weights.squeeze(0).squeeze(0).cpu().data
            loss += self.criterion(decoder_output, targets[di])

            if use_teacher_forcing:
                decoder_input = targets[di]
            else:       # Use the predicted word as the next input
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                predictions.append(ni)
                if ni == vocab.EOS_token:
                    break
                decoder_input = smart_variable(torch.LongTensor([[ni]]))

        return loss, predictions

    def train(input_variable, target_variable):
        self.encoder.train()
        self.decoder.train()
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()

        loss, _ = run_inference(input_variable, target_variable, self.teach_ratio)

        loss.backward()
        if args.grad_clip > 0:
            clip_grad_norm(self.encoder.parameters(), args.grad_clip)
            clip_grad_norm(self.decoder.parameters(), args.grad_clip)
        enc_optimizer.step()
        dec_optimizer.step()

        return loss.data[0] / target_variable.size()[0]

    def validate(input_variable, target_variable, task):
        self.encoder.eval()  # affects the performance of dropout
        self.decoder.eval()

        loss, predictions = run_inference(input_variable, target_variable, teach_ratio=0)

        queries = input_variable.data.tolist()
        targets = target_variable.data.tolist()
        predicted_tokens = [vocab.index_to_word(x, task) for x in predictions]
        query_tokens = [vocab.index_to_word(y[0], task) for y in queries]
        target_tokens = [vocab.index_to_word(z[0], task) for z in targets]

        avg_loss = loss.data[0] / target_variable.size()[0]
        bleu_score = 14 # compute_bleu(predicted_tokens, target_tokens)
        turn_success = [pred == tar[0] for pred, tar in zip(predictions, targets)]

        return avg_loss, bleu_score, all(turn_success)

        # Save model after each epoch
        # print 'Save model checkpoint to', save_path
        # saver.save(sess, save_path, global_step=epoch)

        # Evaluate on dev
        # for split, test_data, num_batches in self.evaluator.dataset():

        #     results = self.eval(sess, split, test_data, num_batches)

        #     # Start to record no improvement epochs
        #     loss = results['loss']
        #     if split == 'dev' and epoch > args.min_epochs:
        #         if loss < best_loss * 0.995:
        #             num_epoch_no_impr = 0
        #         else:
        #             num_epoch_no_impr += 1

        #     if split == 'dev' and loss < best_loss:
        #         print 'New best model'
        #         best_loss = loss
        #         best_saver.save(sess, best_save_path)
        #         self.log_results('best_model', results)
        #         logstats.add('best_model', {'epoch': epoch})

        # # Early stop when no improvement
        # if (epoch > args.min_epochs and num_epoch_no_impr >= 5) or epoch > args.max_epochs:
        #     break
        # epoch += 1

    # def log_results(self, name, results):
    #     logstats.add(name, {'loss': results.get('loss', None)})
    #     logstats.add(name, self.evaluator.log_dict(results))
