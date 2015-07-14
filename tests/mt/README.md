# Example


        head -n1 input | python -m easyhg.mt.decoder --glue-grammar glue --pass-through --weights uniform.ini --count --forest output-exact --rt -v --wp WordPenalty -0.43429466666666666666 --lm LanguageModel 2 btec.klm --viterbi

# cdec output


        INPUT: 一 跳 一 跳 的 痛 。
          id = 0
        Adding pass through grammar
        First pass parse...
          Goal category: [S]
            .......
          Init. forest (nodes/edges): 36/1436
          Init. forest       (paths): 4.44751e+08
          Init. forest  Viterbi logp: 75.9278
          Init. forest       Viterbi: the stomach a jump is forty-five dollars a , please . and jump aches the fall .


          RESCORING PASS #1 [num_fn=2 int_alg=FULL]
          Rescoring forest (full intersection)
            ................................................
          Pass1 forest (nodes/edges): 6252/604238
          Pass1 forest       (paths): 4.44751e+08
          Pass1 forest  Viterbi logp: 36.5898
          Pass1 forest       Viterbi: to jump to jump from temples , please .

        Output kbest to -
        0 ||| to jump to jump from temples , please . ||| LanguageModel=-20.9995 Glue=5 WordPenalty=-3.90865 IsSingletonFE=7 IsSingletonF=2 MaxLexEgivenF=11.9983 MaxLexFgivenE=5.78686 CountEF=2.10721 SampleCountF=13.8385 EgivenFCoherent=13.767 ||| 36.5898


# Interesting settings


Also with `--burn 500`

    time head -n1 input | python -m easyhg.mt.decoder --glue-grammar glue --pass-through --count --forest --rt -v --viterbi --kbest 100 --wp WordPenalty -0.43429466666666666666 --ap Arity 1 output --lm LanguageModel 2 btec.klm2 --framework slice --within ancestral --batch 100 --samples 1000 --weights mira/lm2-p --prior const 1 --temperature0 0.1 --temperature 0.1

    time head -n1 input | python -m easyhg.mt.decoder --glue-grammar glue --pass-through --count --forest --rt -v --viterbi --kbest 100 --wp WordPenalty -0.43429466666666666666 --ap Arity 1 output --lm LanguageModel 2 btec.klm2 --framework slice --within importance --batch 100 --samples 1000 --weights mira/lm2-p --prior const 1 --temperature0 0.1 --temperature 0.1
