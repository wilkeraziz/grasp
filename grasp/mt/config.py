"""
:Authors: - Wilker Aziz
"""
from configparser import ConfigParser
import logging
from ast import literal_eval
from os.path import isfile


def section_literal_eval(items):
    return {k: literal_eval(v) for k, v in items}


def configure(parser, set_defaults=[], required_sections=[], configure_logging=True):
    """
    """
    args = parser.parse_args()

    if configure_logging:
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')


    # parse the config file
    config = ConfigParser()
    # this is necessary in order not to lowercase the keys
    config.optionxform = str


    if args.config:

        if not isfile(args.config):  # TODO write an example file
            raise FileNotFoundError('Could not find your config file: %s' % args.config)

        config.read(args.config)
        # some command line options have default values which may be overwritten by the sections in the config file
        for section in set_defaults:
            if config.has_section(section):
                options = section_literal_eval(config.items(section))
                logging.debug('set_defaults [%s]: %s', section, options)
                parser.set_defaults(**options)

        if set_defaults:
            # reparse options (with new defaults) TODO: find a better way
            args = parser.parse_args()

        # required sections
        failed = False
        for section in required_sections:
            # individual configurations
            if not config.has_section(section):
                logging.error("add a [%s] section to the config file", section)
                failed = True

        if failed:
            raise SyntaxError('Your config is broken.')

    return args, config