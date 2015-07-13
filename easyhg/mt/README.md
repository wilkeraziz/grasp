# Example


        head -n1 input | ~/workspace/github/my-cdec/decoder/cdec -c cdec.ini -w uniform.ini -k 1


        head -n1 input | python -m easyhg.mt.decoder --grammars grammars --glue-grammar glue --pass-through --weights target.ini --lm 2 btec.klm --count --forest -v out-exact
        
        

        time head -n4 input | tail -n1 | python -m easyhg.mt.decoder --glue-grammar glue --pass-through --count --forest --rt -v --weights weights/lm2-p-oov --viterbi --kbest 100 --wp WordPenalty -0.43429466666666666666 --ap Arity 1 output --lm LanguageModel 2 btec.klm --temperature 100 --framework slice --samples 100 --within ancestral --batch 50 --burn 100 --lag 5 --history --experiment n3 --rate 10e5 10e5