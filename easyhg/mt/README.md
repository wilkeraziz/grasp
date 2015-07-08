# Example


        head -n1 input | ~/workspace/github/my-cdec/decoder/cdec -c cdec.ini -w uniform.ini -k 1


        head -n1 input | python -m easyhg.mt.decoder --grammars grammars --glue-grammar glue --pass-through --weights target.ini --lm 2 btec.klm --count --forest -v out-exact