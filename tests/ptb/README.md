# Important parameters


        --grammarfmt discodop
        --log
        --start TOP
        --unkmodel stfd6


# Example


    
        echo 'I was given a million dollars !!!' | python -m easyhg.grammar.parser wsj00 dollars-exact
            --grammarfmt discodop 
            --start TOP 
            --unkmodel stfd6 
            --log 
            --samples 100
            --viterbi 
            --kbest 100



        time echo -e 'I was given -LRB- a million -RRB- dollars !!!' | python -m easyhg.grammar.parser wsj00 dollars-slice 
            --grammarfmt discodop 
            --start TOP 
            --unkmodel stfd6 
            --log 
            --samples 200 
            --framework slice 
            --count 
            --a 0.1 0.2 
            --lag 10 
            --history


# Some interesting settings

        time echo -e 'I was given a million dollars .' | python -m easyhg.grammar.parser wsj00 dollars-exact --grammarfmt discodop --start TOP --unkmodel stfd6 --log --samples 200 --viterbi --kbest 100 --framework slice --prior-a const 0.2 -v

        time echo -e 'I was given a million dollars .' | python -m easyhg.grammar.parser wsj00 dollars-exact --grammarfmt discodop --start TOP --unkmodel stfd6 --log --samples 200 --viterbi --kbest 100 --framework slice --prior-a sym 0.5 -v

# Some not so interesting settings

    time echo -e 'I was given a million dollars .' | python -m easyhg.grammar.parser wsj00 dollars-exact --grammarfmt discodop --start TOP --unkmodel stfd6 --log --samples 200 --viterbi --kbest 100 --framework slice --prior-a beta 2,5 -v
