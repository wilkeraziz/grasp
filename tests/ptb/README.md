# Important parameters


        --grammarfmt discodop
        --log
        --start TOP
        --unkmodel stfd6


# Example


    
        echo 'I was given a million dollars !!!' | python -m easyhg.grammar.parser wsj00 --grammarfmt discodop --start TOP --unkmodel stfd6 --log --samples 10 --viterbi --kbest 10
