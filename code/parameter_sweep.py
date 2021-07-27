import os

from pt_mnist import mnist_pt_objective
from tf_mnist import mnist_tf_objective
from ray.tune.integration.wandb import wandb_mixin
from ray import tune
from argparse import ArgumentParser
import torch
import wandb
import spaceray

@wandb_mixin
def dual_train(config, extra_data_dir):
    # make directory to save weights in
    model_directory = os.path.join(extra_data_dir, 'model_weights/', wandb.run.name)

    pt_test_acc, pt_model, pt_training_history = mnist_pt_objective(config)
    pt_model.eval()
    search_results = {'pt_test_acc': pt_test_acc}
    # save torch model
    torch.save(pt_model.state_dict(), model_directory+'.pt_model.pt')
    # to prevent weird OOM errors
    del pt_model
    torch.cuda.empty_cache()

    tf_test_acc, tf_model, tf_training_history = mnist_tf_objective(config)
    tf_model.save(model_directory+'tf_model')

    search_results['tf_test_acc': tf_test_acc]
    accuracy_diff = abs(pt_test_acc - tf_test_acc)
    # all the logging
    search_results['accuracy_diff'] = accuracy_diff
    search_results['tf_training_loss'] = tf_training_history
    search_results['pt_training_loss'] = pt_training_history
    # log inidividual metrics to wanbd
    for key, value in search_results.items():
        wandb.log({key: value})
    # log custom training and validation curve charts to wandb
    data = [[x, y] for (x, y) in zip(list(range(len(pt_training_history))), pt_training_history)]
    table = wandb.Table(data=data, columns=["epochs", "training_loss"])
    wandb.log({"PT Training Loss": wandb.plot.line(table, "epochs", "training_loss", title="PT Training Loss")})
    data = [[x, y] for (x, y) in zip(list(range(len(tf_training_history))), tf_training_history)]
    table = wandb.Table(data=data, columns=["epochs", "training_loss"])
    wandb.log({"TF Training Loss": wandb.plot.line(table, "epochs", "training_loss", title="TF Training Loss")})
    try:
        tune.report(**search_results)
    except:
        print("Couldn't report Tune results. Continuing.")
        pass
    return search_results


if __name__ == "__main__":
    parser = ArgumentParser("Set output directory, number of trials, and JSON files.")
    parser.add_argument('-t', '--trials', default=25)
    parser.add_argument('-o', '--out', default="results/")
    parser.add_argument('-j', '--json', default="standard.json")
    args = parser.parse_args()
    results = args.out
    os.mkdir(results)
    os.mkdir(os.path.join(results, 'model_weights/'))
    main = os.getcwd()
    results = os.path.join(main, results)
    spaceray.run_experiment(dual_train, args.json, args.trials, args.out, mode="max", metric="accuracy_diff",
                            start_space=0, project_name="mnist_comparison", extra_data_dir=results, num_splits=8,
                            wandb_key="b24709b3f0a9bf7eae4f3a30280c90cd38d1d5f7")
