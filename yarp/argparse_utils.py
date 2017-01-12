import argparse


class CommaSplitAction(argparse.Action):
    '''Split n strip incoming string argument.'''
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [v.strip() for v in values.split(',')])
