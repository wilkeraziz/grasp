# Design

Features are extracted by classes that inherit from `Extractor`.
An extractor is identified by a unique integer and a unique name.

Extractors are responsible for:

    * featurizing an object (e.g. rule, edge)
    * extracting a weight representation from a weight map
    * computing the dot product between the feature and the weight representations
    
Both the feature representation and the weight representation are underspecified, meaning that each extractor can choose 
what to use. Only the dot product is specified, it must return a real number.
    
This allows for a flexible design where the extractor is the only one who knows precisely how its features and weights
are represented. For example, a *word penalty* extractor extracts a single scalar as a feature, thus it needs a single weight.
A *language model* extracts two real-valued features, namely, the LM log probability and an OOV count.
A certain extractor may represent the object using a sparse dictionary.

We have feature extractors of 3 kinds:

* table lookup
* stateless
* stateful

## Table lookup

Extract features from *rules* based on table lookup. Each extractor manages its own tables and the rule itself is the key.

Examples:

    * rule table features

## Stateless

Extract features from *edges* regardless of their context.

Examples:

    * word penalty
    * arity penalty
    

## Stateful

Extract features from *edges* given *context* (a state) and update the context.

Examples:

    * language model
    
### State

The context necessary for a certain feature extractor, also known as *carry*, is often caled *state*.
**Any** hashable object is a potential state. By this simple definition, 
note that an immutable collection of states is also a state. 

This allows for a flexible design where multiple stateful extractors can be combined under a single one which internally
deals with states of different nature mapping them all to a unified representation (such as an integer).


# Model

We combine multiple features under a log-linear model.
The model is aware of the extractors and their weights.

# Scorer

A scorer is an object capable of assigning an edge a score.
It typically contains several feature extractors and a model which combines them.

# Missing

* reordering features

