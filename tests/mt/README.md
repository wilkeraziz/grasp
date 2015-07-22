Example


```bash
time head -n2 input | python -m grasp.mt.decoder --glue-grammar glue --pass-through --rt --wp WordPenalty -0.43429466666666666666 --ap Arity -0.43429466666666666666 --lm LanguageModel 2 btec.klm2 --weights mira/lm2-p --temperature 0.1 --samples 1000 --framework slice --batch 100 --within uniform --temperature0 10 --prior const 1 output --experiment uniform --progress
python -m grasp.mt.decision output/uniform/slice/yields --rule consensus output/uniform/slice/yields/consensus
```


For more check the documentation in `grasp.mt`.
            