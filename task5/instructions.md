generate code in adversarial.ipynb to attack the models trained in task4 and task1.
load the weights of the strong robust model in task4 from ../task4/TEA/outputs/teacher_weights.pt
weights of the weak model from task1 in ../task1/saved_models/cnn_weights_feb1_GODLYPULL.pth

1. evaluate the performance of both models on datasets in ../task0/outputs/colored-mnist/ (train and test)
2. generate adversarial examples using FGSM attack on both models and evaluate their performance on the adversarial examples.
 ie no regularization, no defense, just pure FGSM attack.
3. compare the results and analyze the differences in performance between the two models on clean and adversarial examples.
To perform the tasks outlined, you can follow the steps below in your `adversarial.ipynb` notebook. This code assumes you have the necessary libraries installed (like PyTorch) and that you have access to the datasets and model weights as specified. 


Task 5: The Invisible Cloak
"Reality is merely an illusion, albeit a very persistent one." — Einstein

Take your robust model. Can you perform a Targeted Adversarial Attack?

Take an image of a 7.
Optimize a noise pattern (perturbation) such that the model predicts a 3 with >90% confidence.
The Twist: The perturbation must be invisible to the human eye (Constraint: Max pixel change epsilon < 0.05).
The Question: Is your "robust" model (which ignores color) harder to fool than your "lazy" model from Task 1? Quantify the difference in the required noise magnitude.



take 10 images of 7 randomly from test set and perform FGSM attack to make the model predict 3 with >90% confidence.
take 10 images of 7 randomly from test set and perform targeted attack to make the model predict 3 with >90% confidence. 

Compare the results between the robust model and the lazy model. Quantify the difference in the required noise magnitude for both models.