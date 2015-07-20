# Example


        head -n1 input | ~/workspace/github/my-cdec/decoder/cdec -c cdec.ini -w uniform.ini -k 1


        head -n1 input | python -m easyhg.mt.decoder --grammars grammars --glue-grammar glue --pass-through --weights target.ini --lm 2 btec.klm --count --forest -v out-exact
        
        

        time head -n4 input | tail -n1 | python -m easyhg.mt.decoder --glue-grammar glue --pass-through --count --forest --rt -v --weights weights/lm2-p-oov --viterbi --kbest 100 --wp WordPenalty -0.43429466666666666666 --ap Arity 1 output --lm LanguageModel 2 btec.klm --temperature 100 --framework slice --samples 100 --within ancestral --batch 50 --burn 100 --lag 5 --history --experiment n3 --rate 10e5 10e5
        
        time head input | python -m easyhg.mt.decoder --glue-grammar glue --pass-through --count --forest --rt -v --viterbi --kbest 100 --wp WordPenalty -0.43429466666666666666 --ap Arity 1 output --lm LanguageModel 2 btec.klm2 --framework slice --burn 1000 --within importance --batch 100 --experiment debug-90 --samples 1000 --weights mira/lm2-p --prior const 1 --temperature0 0.1 --temperature 0.1
        
        cat evaldev/dev200.input | python -m easyhg.mt.decoder --glue-grammar glue --pass-through --count --forest --rt --wp WordPenalty -0.43429466666666666666 --ap Arity -0.43429466666666666666 --lm LanguageModel 2 btec.klm2 --weights mira/lm2-p --temperature 0.1 --temperature0 10 --framework slice --within importance --batch 50 --samples 2000 --prior const 1 --cpus 16 --save-chain evaldev/lm2 --experiment 1 2> evaldev/log.1