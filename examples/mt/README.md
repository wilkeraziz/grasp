Example


```bash
time head -n2 input | python -m grasp.mt.decoder --glue-grammar glue --pass-through --rt --wp WordPenalty -0.43429466666666666666 --ap Arity -0.43429466666666666666 --lm LanguageModel 2 btec.klm2 --weights mira/lm2-p --temperature 0.1 --samples 1000 --framework slice --batch 100 --within uniform --temperature0 10 --prior const 1 output --experiment uniform --progress
python -m grasp.mt.decision output/uniform/slice/yields --rule consensus output/uniform/slice/yields/consensus
```


For more check the documentation in `grasp.mt`.
            


# MC 
    time head -n1 input | python -m grasp.mt.cydecoder --grammars grammars --glue-grammar glue --pass-through --rt --lm LanguageModel 2 btec.klm2 --weights mira/lm2-p --temperature 0.1  output --forest -v --wp WordPenalty 1 --ap Arity 1 --framework slice --prior const 1  --within importance --batch 100 --samples 200 --save-chain --lag 1 --burn 100 --experiment MC-3 --viterbi -v --prior sym 0.1 --framework exact --samples 1000

# MCMC
    time head -n1 input | python -m grasp.mt.cydecoder --grammars grammars --glue-grammar glue --pass-through --rt --lm LanguageModel 2 btec.klm2 --weights mira/lm2-p --temperature 0.1  output --forest -v --wp WordPenalty 1 --ap Arity 1 --framework slice --prior const 1  --within importance --batch 100 --samples 200 --save-chain --lag 1 --burn 100 --experiment MCMC-importance-3 --viterbi -v --prior sym 0.1


# minrisk

time python -m grasp.mt.minrisk MinRisk chisel-minrisk/proxy.ini chisel-minrisk/target.ini head --glue-grammar glue --pass-through --experiment debug-dev2-head -v --devtest dev2-head --proxy-weights chisel-minrisk/proxy.weights --target-weights chisel-minrisk/target.weights --samples 1000  --devtest-grammars chisel-minrisk/dev2 -Tp 10 -Tq 0.1
