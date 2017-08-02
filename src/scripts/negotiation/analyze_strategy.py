from collections import defaultdict
import json
import utils
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from src.basic.negotiation.price_tracker import PriceTracker, PriceScaler, add_price_tracker_arguments
from src.basic.negotiation.tokenizer import tokenize
from src.basic.scenario_db import NegotiationScenario
from src.basic.entity import Entity
import nltk.data
import re

__author__ = 'anushabala'


THRESHOLD = 30.0
MAX_MARGIN = 2.4
MIN_MARGIN = -2.0


def round_partial(value, resolution=0.1):
    return round (value / resolution) * resolution


class SpeechActs(object):
    GEN_QUESTION = 'general_question'
    PRICE_QUESTION = 'price_request'
    GEN_STATEMENT = 'general_statement'
    PRICE_STATEMENT = 'price_statement'
    GREETING = 'greeting'
    AGREEMENT = 'agreement'

    ACTS = [GEN_QUESTION, PRICE_QUESTION, GEN_STATEMENT, PRICE_STATEMENT, GREETING, AGREEMENT]


class SpeechActAnalyzer(object):
    agreement_patterns = [
        r'that works',
        r'i could do',
        r'[^a-zA-Z]ok[^a-zA-Z]|okay',
        r'great'
    ]

    @classmethod
    def is_question(cls, tokens):
        last_word = tokens[-1]
        first_word = tokens[0]
        return last_word == '?' or first_word in ('how', 'do', 'does', 'are', 'is', 'what', 'would', 'will')

    @classmethod
    def get_question_type(cls, tokens):
        if not cls.is_question(tokens):
            return None
        if cls.is_price_statement(tokens):
            return SpeechActs.PRICE_QUESTION
        return SpeechActs.GEN_QUESTION

    @classmethod
    def is_agreement(cls, raw_sentence):
        for pattern in cls.agreement_patterns:
            if re.match(pattern, raw_sentence, re.IGNORECASE) is not None:
                return True
        return False

    @classmethod
    def is_price_statement(cls, tokens):
        for token in tokens:
            if isinstance(token, Entity) and token.canonical.type == 'price':
                return True
            else:
                return False

    @classmethod
    def is_greeting(cls, tokens):
        for token in tokens:
            if token in ('hi', 'hello', 'hey', 'hiya', 'howdy'):
                return True
        return False

    @classmethod
    def get_speech_act(cls, sentence, linked_tokens):
        if cls.is_question(linked_tokens):
            return cls.get_question_type(linked_tokens)
        if cls.is_price_statement(linked_tokens):
            return SpeechActs.PRICE_STATEMENT
        if cls.is_agreement(sentence):
            return SpeechActs.AGREEMENT
        if cls.is_greeting(linked_tokens):
            return SpeechActs.GREETING

        return SpeechActs.GEN_STATEMENT


class StrategyAnalyzer(object):
    nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def __init__(self, transcripts_path, stats_path, price_tracker_model, debug=False):
        transcripts = json.load(open(transcripts_path, 'r'))
        if debug:
            transcripts = transcripts[:100]
        self.dataset = utils.filter_rejected_chats(transcripts)

        self.price_tracker = PriceTracker(price_tracker_model)

        # group chats depending on whether the seller or the buyer wins
        self.buyer_wins, self.seller_wins = self.group_outcomes_and_roles()

        self.stats_path = stats_path
        if not os.path.exists(self.stats_path):
            os.makedirs(self.stats_path)

    @classmethod
    def get_price_trend(cls, price_tracker, chat, agent=None):
        def _normalize_price(seen_price):
            return (float(seller_target) - float(seen_price)) / (float(seller_target) - float(buyer_target))

        scenario = NegotiationScenario.from_dict(None, chat['scenario'])
        # print chat['scenario']
        kbs = scenario.kbs
        roles = {
            kbs[0].facts['personal']['Role']: 0,
            kbs[1].facts['personal']['Role']: 1
        }

        buyer_target = kbs[roles[utils.BUYER]].facts['personal']['Target']
        seller_target = kbs[roles[utils.SELLER]].facts['personal']['Target']

        prices = []
        for e in chat['events']:
            if e['action'] == 'message':
                if agent is not None and e['agent'] != agent:
                    continue
                raw_tokens = tokenize(e['data'])
                # link entity
                linked_tokens = price_tracker.link_entity(raw_tokens,
                                                               kb=kbs[e['agent']])
                for token in linked_tokens:
                    if isinstance(token, Entity):
                        try:
                            replaced = PriceScaler.unscale_price(kbs[e['agent']], token)
                        except OverflowError:
                            print "Raw tokens: ", raw_tokens
                            print "Overflow error: {:s}".format(token)
                            print kbs[e['agent']].facts
                            print "-------"
                            continue
                        norm_price = _normalize_price(replaced.canonical.value)
                        if 0. <= norm_price <= 2.:
                            # if the number is greater than the list price or significantly lower than the buyer's
                            # target it's probably not a price
                            prices.append(norm_price)
                # do some stuff here
            elif e['action'] == 'offer':
                norm_price = _normalize_price(e['data']['price'])
                if 0. <= norm_price <= 2.:
                    prices.append(norm_price)
                # prices.append(e['data']['price'])

        # print "Chat: {:s}".format(chat['uuid'])
        # print "Trend:", prices

        return prices

    @classmethod
    def split_turn(cls, turn):
        # a single turn can be comprised of multiple sentences
        return cls.nltk_tokenizer.tokenize(turn)

    @classmethod
    def get_speech_acts(cls, chat, price_tracker, agent=None, role=None):
        scenario = NegotiationScenario.from_dict(None, chat['scenario'])
        kbs = scenario.kbs
        roles = {
            kbs[0].facts["personal"]["Role"]: 0,
            kbs[1].facts["personal"]["Role"]: 1
        }
        # print chat['scenario']

        acts = []
        for e in chat['events']:
            if e['action'] != 'message':
                continue
            if agent is not None and e['agent'] != agent:
                continue
            if role is not None and roles[role] != e['agent']:
                continue

            sentences = cls.split_turn(e['data'])

            for s in sentences:
                tokens = tokenize(s)
                linked_tokens = price_tracker.link_entity(tokens, kb=kbs[e['agent']])
                acts.append(SpeechActAnalyzer.get_speech_act(s, linked_tokens))

        return acts

    def group_outcomes_and_roles(self):
        buyer_wins = []
        seller_wins = []
        ties = 0
        total_chats = 0
        for ex in self.dataset:
            roles = {0: ex["scenario"]["kbs"][0]["personal"]["Role"],
                     1: ex["scenario"]["kbs"][1]["personal"]["Role"]}
            winner = utils.get_winner(ex)
            if winner is None:
                continue
            total_chats += 1
            if winner == -1:
                buyer_wins.append(ex)
                seller_wins.append(ex)
                ties += 1
            elif roles[winner] == utils.BUYER:
                buyer_wins.append(ex)
            elif roles[winner] == utils.SELLER:
                seller_wins.append(ex)

        print "# of ties: {:d}".format(ties)
        print "Total chats with outcomes: {:d}".format(total_chats)
        return buyer_wins, seller_wins

    def plot_length_vs_margin(self, out_name='turns_vs_margin.png'):
        labels = ['buyer wins', 'seller wins']
        plt.figure(figsize=(10, 6))

        for (chats, lbl) in zip([self.buyer_wins, self.seller_wins], labels):
            margins = defaultdict(list)
            for ex in chats:
                turns = utils.get_turns_per_agent(ex)
                total_turns = turns[0] + turns[1]
                margin = utils.get_margin(ex)
                if margin > MAX_MARGIN or margin < 0.:
                    continue

                margins[total_turns].append(margin)

            sorted_keys = list(sorted(margins.keys()))

            turns = []
            means = []
            errors = []
            for k in sorted_keys:
                if len(margins[k]) >= THRESHOLD:
                    turns.append(k)
                    means.append(np.mean(margins[k]))
                    errors.append(stats.sem(margins[k]))

            plt.errorbar(turns, means, yerr=errors, label=lbl, fmt='--o')

        plt.legend()
        plt.xlabel('# of turns in dialogue')
        plt.ylabel('Margin of victory')

        save_path = os.path.join(self.stats_path, out_name)
        plt.savefig(save_path)

    def plot_margin_histograms(self):
        for (lbl, group) in zip(['buyer_wins', 'seller_wins'], [self.buyer_wins, self.seller_wins]):
            margins = []
            for ex in group:
                winner = utils.get_winner(ex)
                if winner is None:
                    continue
                margin = utils.get_margin(ex)
                if 0 <= margin <= MAX_MARGIN:
                    margins.append(margin)

            b = np.linspace(0, MAX_MARGIN, num=int(MAX_MARGIN/0.2)+2)
            print b
            hist, bins = np.histogram(margins, bins=b)

            width = np.diff(bins)
            center = (bins[:-1] + bins[1:]) / 2

            fig, ax = plt.subplots(figsize=(8,3))
            ax.bar(center, hist, align='center', width=width)
            ax.set_xticks(bins)

            save_path = os.path.join(self.stats_path, '{:s}_wins_margins_histogram.png'.format(lbl))
            plt.savefig(save_path)

    def plot_length_histograms(self):
        lengths = []
        for ex in self.dataset:
            winner = utils.get_winner(ex)
            if winner is None:
                continue
            turns = utils.get_turns_per_agent(ex)
            total_turns = turns[0] + turns[1]
            lengths.append(total_turns)

        hist, bins = np.histogram(lengths)

        width = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2

        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(center, hist, align='center', width=width)
        ax.set_xticks(bins)

        save_path = os.path.join(self.stats_path, 'turns_histogram.png')
        plt.savefig(save_path)

    def plot_price_trends(self, top_n=10):
        labels = ['buyer_wins', 'seller_wins']
        for (group, lbl) in zip([self.buyer_wins, self.seller_wins], labels):
            plt.figure(figsize=(10, 6))
            trends = []
            for chat in group:
                winner = utils.get_winner(chat)
                margin = utils.get_margin(chat)
                if margin > 1.0 or margin < 0.:
                    continue
                if winner is None:
                    continue

                # print "Winner: Agent {:d}\tWin margin: {:.2f}".format(winner, margin)
                if winner == -1 or winner == 0:
                    trend = self.get_price_trend(self.price_tracker, chat, agent=0)
                    if len(trend) > 1:
                        trends.append((margin, chat, trend))
                if winner == -1 or winner == 1:
                    trend = self.get_price_trend(self.price_tracker, chat, agent=1)
                    if len(trend) > 1:
                        trends.append((margin, chat,  trend))

                # print ""

            sorted_trends = sorted(trends, key=lambda x:x[0], reverse=True)
            for (idx, (margin, chat, trend)) in enumerate(sorted_trends[:top_n]):
                print '{:s}: Chat {:s}\tMargin: {:.2f}'.format(lbl, chat['uuid'], margin)
                print 'Trend: ', trend
                print chat['scenario']['kbs']
                print ""
                plt.plot(trend, label='Margin={:.2f}'.format(margin))
            plt.legend()
            plt.xlabel('N-th price mentioned in chat')
            plt.ylabel('Value of mentioned price')
            out_path = os.path.join(self.stats_path, '{:s}_trend.png'.format(lbl))
            plt.savefig(out_path)

    def _get_price_mentions(self, chat, agent=None):
        scenario = NegotiationScenario.from_dict(None, chat['scenario'])
        # print chat['scenario']
        kbs = scenario.kbs

        prices = 0
        for e in chat['events']:
            if agent is not None and e['agent'] != agent:
                    continue
            if e['action'] == 'message':
                raw_tokens = tokenize(e['data'])
                # link entity
                linked_tokens = self.price_tracker.link_entity(raw_tokens,
                                                               kb=kbs[e['agent']])
                for token in linked_tokens:
                    if isinstance(token, Entity) and token.canonical.type == 'price':
                        prices += 1

        return prices

    def plot_speech_acts(self):
        labels = ['buyer_wins', 'seller_wins']
        for (group, lbl) in zip([self.buyer_wins, self.seller_wins], labels):
            plt.figure(figsize=(10, 6))
            speech_act_counts = dict((act, defaultdict(list)) for act in SpeechActs.ACTS)
            for chat in group:
                winner = utils.get_winner(chat)
                margin = utils.get_margin(chat)
                if margin > MAX_MARGIN or margin < 0.:
                    continue
                if winner is None:
                    continue

                margin = round_partial(margin) # round the margin to the nearest 0.1 to reduce noise

                if winner == -1 or winner == 0:
                    speech_acts = self.get_speech_acts(chat, agent=0)
                    # print "Chat {:s}\tWinner: {:d}".format(chat['uuid'], winner)
                    # print speech_acts
                    for act in SpeechActs.ACTS:
                        frac = float(speech_acts.count(act))/float(len(speech_acts))
                        speech_act_counts[act][margin].append(frac)
                if winner == -1 or winner == 1:
                    speech_acts = self.get_speech_acts(chat, agent=1)
                    # print "Chat {:s}\tWinner: {:d}".format(chat['uuid'], winner)
                    # print speech_acts
                    for act in SpeechActs.ACTS:
                        frac = float(speech_acts.count(act))/float(len(speech_acts))
                        speech_act_counts[act][margin].append(frac)

            for act in SpeechActs.ACTS:
                counts = speech_act_counts[act]
                margins = []
                fracs = []
                errors = []
                bin_totals = 0.
                for m in sorted(counts.keys()):
                    if len(counts[m]) > THRESHOLD:
                        bin_totals += len(counts[m])
                        margins.append(m)
                        fracs.append(np.mean(counts[m]))
                        errors.append(stats.sem(counts[m]))
                print bin_totals / float(len(margins))

                plt.errorbar(margins, fracs, yerr=errors, label=act, fmt='--o')

            plt.xlabel('Margin of victory')
            plt.ylabel('Fraction of speech act occurences')
            plt.title('Speech act frequency vs. margin of victory')
            plt.legend()
            save_path = os.path.join(self.stats_path, '{:s}_speech_acts.png'.format(lbl))
            plt.savefig(save_path)

    def plot_speech_acts_by_role(self):
        labels = utils.ROLES
        for lbl in labels:
            plt.figure(figsize=(10, 6))
            speech_act_counts = dict((act, defaultdict(list)) for act in SpeechActs.ACTS)
            for chat in self.dataset:
                if utils.get_winner(chat) is None:
                    # skip chats with no outcomes
                    continue
                speech_acts = self.get_speech_acts(chat, role=lbl)
                agent = 1 if chat['scenario']['kbs'][1]['personal']['Role'] == lbl else 0
                margin = utils.get_margin(chat, agent=agent)
                if margin > MAX_MARGIN:
                    continue
                margin = round_partial(margin)
                for act in SpeechActs.ACTS:
                    frac = float(speech_acts.count(act))/float(len(speech_acts))
                    speech_act_counts[act][margin].append(frac)

            for act in SpeechActs.ACTS:
                counts = speech_act_counts[act]
                margins = []
                fracs = []
                errors = []
                for m in sorted(counts.keys()):
                    if len(counts[m]) > THRESHOLD:
                        margins.append(m)
                        fracs.append(np.mean(counts[m]))
                        errors.append(stats.sem(counts[m]))

                plt.errorbar(margins, fracs, yerr=errors, label=act, fmt='--o')

            plt.xlabel('Margin of victory')
            plt.ylabel('Fraction of speech act occurences')
            plt.title('Speech act frequency vs. margin of victory')
            plt.legend()
            save_path = os.path.join(self.stats_path, '{:s}_speech_acts.png'.format(lbl))
            plt.savefig(save_path)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='Directory containing all output from website')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (only run on 50 chats)')
    add_price_tracker_arguments(parser)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        raise ValueError("Output directory {:s} doesn't exist".format(args.output_dir))

    transcripts_path = os.path.join(args.output_dir, 'transcripts', 'transcripts.json')
    stats_output = os.path.join(args.output_dir, 'stats')

    analyzer = StrategyAnalyzer(transcripts_path, stats_output, args.price_tracker_model, args.debug)
    # analyzer.plot_length_histograms()
    # analyzer.plot_margin_histograms()
    # analyzer.plot_length_vs_margin()
    # analyzer.plot_price_trends()
    analyzer.plot_speech_acts()
    analyzer.plot_speech_acts_by_role()
