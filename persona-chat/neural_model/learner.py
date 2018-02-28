import numpy as np

from cocoa.lib import logstats
# from cocoa.model.learner import Learner as BaseLearner, add_learner_arguments
from cocoa.lib.bleu import compute_bleu

# from model.ranker import EncDecRanker

def add_learner_arguments(parser):
    parser.add_argument('--optimizer', default='sgd', help='Optimization method')
    parser.add_argument('--sample-targets', action='store_true', help='Sample targets from candidates')
    parser.add_argument('--grad-clip', type=int, default=5, help='Min and max values of gradients')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--min-epochs', type=int, default=10, help='Number of training epochs to run before checking for early stop')
    parser.add_argument('--max-epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--num-per-epoch', type=int, default=None, help='Number of examples per epoch')
    parser.add_argument('--print-every', type=int, default=1, help='Number of examples between printing training loss')
    parser.add_argument('--init-from', help='Initial parameters')
    parser.add_argument('--checkpoint', default='.', help='Directory to save learned models')
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU or not')
    parser.add_argument('--summary-dir', default='/tmp', help='Path to summary logs')
    parser.add_argument('--eval-modes', nargs='*', default=('loss',), help='What to evaluate {loss, generation}')

optim = {'adagrad': optim.Adagrad,
         'sgd': optim.SGD,
         'adam': optim.Adam,
        }

class Learner(object):
    def __init__(self, data, model, evaluator, batch_size=1, summary_dir='/tmp', verbose=False):
        self.data = data  # DataGenerator object
        self.model = model
        self.vocab = data.mappings['vocab']
        self.batch_size = batch_size
        self.evaluator = evaluator
        self.verbose = verbose
        self.summary_dir = summary_dir

    def _run_batch(self, dialogue_batch, test=False):
        '''
        Run truncated RNN through a sequence of batch examples.
        '''
        for batch in dialogue_batch['batch_seq']:
            # TODO: hacky
            if init_price_history is None and hasattr(self.model.decoder, 'price_predictor'):
                batch_size = batch['encoder_inputs'].shape[0]
                init_price_history = self.model.decoder.price_predictor.zero_init_price(batch_size)
            feed_dict = self._get_feed_dict(batch, encoder_init_state, test=test, init_price_history=init_price_history)
            fetches = {
                    'loss': self.model.loss,
                    }

            if self.model.name == 'encdec':
                fetches['raw_preds'] = self.model.decoder.output_dict['logits']
            elif self.model.name == 'selector':
                fetches['raw_preds'] = self.model.decoder.output_dict['scores']
            else:
                raise ValueError

            if not test:
                fetches['train_op'] = self.train_op
                fetches['gn'] = self.grad_norm
            else:
                fetches['total_loss'] = self.model.total_loss

            if self.model.stateful:
                fetches['final_state'] = self.model.final_state

            if not test:
                self.global_step += 1
                if self.global_step % 100 == 0:
                    self.train_writer.add_summary(results['merged'], self.global_step)

            # if self.verbose:
            #     preds = self.model.output_to_preds(results['raw_preds'])
            #     self._print_batch(batch, preds, results['loss'])


            # TODO: refactor
            if self.model.name == 'selector':
                labels = batch['decoder_args']['candidate_labels']
                preds = results['raw_preds']
                for k in (1, 5):
                    recall = self.evaluator.recall_at_k(labels, preds, k=k, summary_map=summary_map)
                    logstats.update_summary_map(summary_map, {'recall_at_{}'.format(k): recall})

    def test_loss(self, sess, test_data, num_batches):
        # Return the cross-entropy loss.
        summary_map = {}
        for i in xrange(num_batches):
            dialogue_batch = test_data.next()
            self._run_batch(dialogue_batch, sess, summary_map, test=True)
        return summary_map

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

    def learn(args, config, encoder, decoder, sources, targets, criterion):
      loss = 0
      encoder_hidden = encoder.initHidden()
      encoder_length = sources.size()[0]
      encoder_outputs, encoder_hidden = encoder(sources, encoder_hidden)

      decoder_hidden = encoder_hidden
      decoder_length = targets.size()[0]
      decoder_input = smart_variable(torch.LongTensor([[vocab.SOS_token]]))
      decoder_context = smart_variable(torch.zeros(1, 1, decoder.hidden_size))

      visual = torch.zeros(encoder_length, decoder_length)
      predictions = []
      for di in range(decoder_length):
        use_teacher_forcing = random.random() < args.teach_ratio
        if decoder.expand_params:
          decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_outputs,
            sources, targets, di, use_teacher_forcing)
        else:
          decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
              decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        visual[:, di] = attn_weights.squeeze(0).squeeze(0).cpu().data
        loss += criterion(decoder_output, targets[di])

        if use_teacher_forcing:
          decoder_input = targets[di]
        else:       # Use the predicted word as the next input
          topv, topi = decoder_output.data.topk(1)
          ni = topi[0][0]
          predictions.append(ni)
          if ni == vocab.EOS_token:
            break
          decoder_input = smart_variable(torch.LongTensor([[ni]]))

      return loss, predictions, visual

    def clip_gradient(models, clip):
      '''
      models: a list, such as [encoder, decoder]
      clip: amount to clip the gradients by
      '''
      if clip is None:
        return
      for model in models:
        clip_grad_norm(model.parameters(), clip)

    def learn(self, args, config, split='train'):
        # logstats.init(stats_file)
        assert args.min_epochs <= args.max_epochs
        assert args.optimizer in optim.keys()
        optimizer = optim[args.optimizer](args.learning_rate)

        # Gradient
        grads_and_vars = optimizer.compute_gradients(self.model.loss)
        if args.grad_clip > 0:
            min_grad, max_grad = -1.*args.grad_clip, args.grad_clip
            clipped_grads_and_vars = [
                (tf.clip_by_value(grad, min_grad, max_grad) if grad is not None else grad, var) \
                for grad, var in grads_and_vars]
        else:
            clipped_grads_and_vars = grads_and_vars
        self.grad_norm = tf.global_norm([grad for grad, var in grads_and_vars])
        self.clipped_grad_norm = tf.global_norm([grad for grad, var in clipped_grads_and_vars])
        self.grad_norm = self.clipped_grad_norm

        # Optimize
        self.train_op = optimizer.apply_gradients(clipped_grads_and_vars)

        # Training loop
        train_data = self.data.generator(split, self.batch_size)
        num_per_epoch = train_data.next()
        step = 0
        saver = tf.train.Saver()
        save_path = os.path.join(args.checkpoint, 'tf_model.ckpt')
        best_saver = tf.train.Saver(max_to_keep=1)
        best_checkpoint = args.checkpoint+'-best'
        if not os.path.isdir(best_checkpoint):
            os.mkdir(best_checkpoint)
        best_save_path = os.path.join(best_checkpoint, 'tf_model.ckpt')
        best_loss = float('inf')
        # Number of iterations without any improvement
        num_epoch_no_impr = 0
        self.global_step = 0

        # Testing
        with tf.Session(config=config) as sess:
            # Summary
            self.merged_summary = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())
            if args.init_from:
                saver.restore(sess, ckpt.model_checkpoint_path)
            summary_map = {}
            epoch = 1
            while True:
                print '================== Epoch %d ==================' % (epoch)
                for i in xrange(num_per_epoch):
                    start_time = time.time()
                    self._run_batch(train_data.next(), sess, summary_map, test=False)
                    end_time = time.time()
                    results = self.collect_summary_train(summary_map)
                    results['time(s)/batch'] = end_time - start_time
                    results['memory(MB)'] = memory()
                    results_str = ' '.join(['{}={:.4f}'.format(k, v) for k, v in sorted(results.items())])
                    step += 1
                    if step % args.print_every == 0 or step % num_per_epoch == 0:
                        print '{}/{} (epoch {}) {}'.format(i+1, num_per_epoch, epoch, results_str)
                        summary_map = {}  # Reset
                step = 0

                # Save model after each epoch
                print 'Save model checkpoint to', save_path
                saver.save(sess, save_path, global_step=epoch)

                # Evaluate on dev
                for split, test_data, num_batches in self.evaluator.dataset():

                    results = self.eval(sess, split, test_data, num_batches)

                    # Start to record no improvement epochs
                    loss = results['loss']
                    if split == 'dev' and epoch > args.min_epochs:
                        if loss < best_loss * 0.995:
                            num_epoch_no_impr = 0
                        else:
                            num_epoch_no_impr += 1

                    if split == 'dev' and loss < best_loss:
                        print 'New best model'
                        best_loss = loss
                        best_saver.save(sess, best_save_path)
                        self.log_results('best_model', results)
                        logstats.add('best_model', {'epoch': epoch})

                # Early stop when no improvement
                if (epoch > args.min_epochs and num_epoch_no_impr >= 5) or epoch > args.max_epochs:
                    break
                epoch += 1

    def log_results(self, name, results):
        logstats.add(name, {'loss': results.get('loss', None)})
        logstats.add(name, self.evaluator.log_dict(results))