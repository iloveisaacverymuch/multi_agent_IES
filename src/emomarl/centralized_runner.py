import os
import time
from copy import deepcopy

import numpy as np

from src.emomarl.centralized_conditioned_masac import (
    build_dense_preference_specs,
    initialize_shared_conditioned_learner,
    run_conditioned_centralized_generation,
)
from src.emomarl.conditioned_masac import ConditionedMASACReplayBuffer
from src.emomarl.ep import EP
from src.emomarl.opt_graph import OptGraph
from src.emomarl.population import Population
from src.emomarl.runner import save_final_ep, select_diverse_masac_tasks
from src.emomarl.task import Task
from src.emomarl.warm_up import initialize_warm_up_batch


def _write_objs(path, objs_batch, obj_num):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "objs.txt"), "w") as fp:
        for objs in objs_batch:
            fp.write(("{:5f}" + (obj_num - 1) * ",{:5f}" + "\n").format(*objs))


def _write_opt_graph(path, opt_graph, population, obj_num):
    with open(os.path.join(path, "optgraph.txt"), "w") as fp:
        fp.write("{}\n".format(len(opt_graph.objs)))
        for i in range(len(opt_graph.objs)):
            fp.write(
                ("{:5f}" + (obj_num - 1) * ",{:5f}" + ";"
                 + "{:5f}" + (obj_num - 1) * ",{:5f}" + ";{}\n").format(
                    *(opt_graph.weights[i]), *(opt_graph.objs[i]), opt_graph.prev[i]
                )
            )
        fp.write("{}\n".format(len(population.sample_batch)))
        for sample in population.sample_batch:
            fp.write("{}\n".format(sample.optgraph_id))


def _write_elites(path, args, elite_batch, weight_batch, predicted_offspring_objs, all_offspring_batch):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "elites.txt"), "w") as fp:
        for elite in elite_batch:
            fp.write(("{:5f}" + (args.obj_num - 1) * ",{:5f}" + "\n").format(*(elite.objs)))

    with open(os.path.join(path, "weights.txt"), "w") as fp:
        for weight in weight_batch:
            fp.write(("{:5f}" + (args.obj_num - 1) * ",{:5f}" + "\n").format(*(weight)))

    if args.selection_method in (
        "prediction-guided",
        "contribution-guided",
        "coverage-contribution-guided",
        "uncertainty-contribution-guided",
        "ensemble-uncertainty-guided",
        "ensemble-uncertainty-extreme-guided",
        "hv-ensemble-guided",
    ):
        with open(os.path.join(path, "predictions.txt"), "w") as fp:
            for objs in predicted_offspring_objs:
                fp.write(("{:5f}" + (args.obj_num - 1) * ",{:5f}" + "\n").format(*(objs)))

    with open(os.path.join(path, "offsprings.txt"), "w") as fp:
        for offsprings in all_offspring_batch:
            for sample in offsprings:
                fp.write(
                    ("{:5f}" + (args.obj_num - 1) * ",{:5f}" + "\n").format(*(sample.objs))
                )


def run_conditioned_centralized(args):
    run_start_time = time.time()
    total_generation = args.total_generation
    ep = EP()
    population = Population(args)
    opt_graph = OptGraph()

    elite_batch, weight_batch = initialize_warm_up_batch(args)
    for sample, weight in zip(elite_batch, weight_batch):
        sample.optgraph_id = opt_graph.insert(deepcopy(weight), deepcopy(sample.objs), -1)

    use_shared_learner = getattr(args, "conditioned_shared_learner", True)
    shared_agent = None
    shared_replay_buffer = None
    shared_env_steps = 0
    if use_shared_learner:
        shared_agent = initialize_shared_conditioned_learner(args, elite_batch)
        shared_replay_buffer = ConditionedMASACReplayBuffer(args)

    generation_now = 0
    while generation_now < total_generation:
        if generation_now == 0:
            num_updates = args.warmup_iter
            print("\n------------------------------- 热身阶段 -------------------------------")
        else:
            num_updates = args.update_iter
            print(f"\n-------------------- 进化阶段: 第 {generation_now} 代 --------------------")
        generation_now += 1

        task_batch = [Task(elite, weight) for elite, weight in zip(elite_batch, weight_batch)]
        eval_specs = build_dense_preference_specs(args, task_batch)
        (
            shared_agent,
            shared_replay_buffer,
            shared_env_steps,
            all_offspring_batch,
            eval_specs,
        ) = run_conditioned_centralized_generation(
            args,
            task_batch,
            num_updates,
            generation_now,
            agent=shared_agent if use_shared_learner else None,
            replay_buffer=shared_replay_buffer if use_shared_learner else None,
            env_steps=shared_env_steps if use_shared_learner else 0,
            eval_specs=eval_specs,
        )
        if not use_shared_learner:
            shared_agent = None
            shared_replay_buffer = None
            shared_env_steps = 0

        all_sample_batch = []
        offspring_batch = []
        for spec_id, offsprings in enumerate(all_offspring_batch):
            if len(offsprings) == 0:
                print(f"WARNING: eval spec {spec_id} produced no offspring; skipping it.")
                continue
            spec = eval_specs[spec_id]
            prev_node_id = spec.get("parent_optgraph_id")
            opt_weights = deepcopy(spec["weight"])

            for i, sample in enumerate(offsprings):
                all_sample_batch.append(sample)
                if (i + 1) % args.sample_iter == 0:
                    prev_node_id = opt_graph.insert(opt_weights, deepcopy(sample.objs), prev_node_id)
                    sample.optgraph_id = prev_node_id
                    offspring_batch.append(sample)

        ep.update(all_sample_batch)
        population.update(offspring_batch)

        if generation_now == total_generation:
            elite_batch, weight_batch, predicted_offspring_objs = [], [], []
            print("Final generation completed; skipping next-task selection.")
        elif args.masac_diverse_task_selection:
            elite_batch, weight_batch, predicted_offspring_objs = select_diverse_masac_tasks(
                args, ep, population
            )
        else:
            elite_batch, weight_batch, predicted_offspring_objs = (
                population.prediction_guided_selection(args, ep, opt_graph)
            )

        if generation_now != total_generation:
            print("Selected Tasks:")
            for i in range(len(elite_batch)):
                print("objs = {}, weight = {}".format(elite_batch[i].objs, weight_batch[i]))

        ep_dir = os.path.join(args.save_dir, str(generation_now), "ep")
        _write_objs(ep_dir, ep.obj_batch, args.obj_num)

        population_dir = os.path.join(args.save_dir, str(generation_now), "population")
        _write_objs(population_dir, [sample.objs for sample in population.sample_batch], args.obj_num)
        _write_opt_graph(population_dir, opt_graph, population, args.obj_num)

        elite_dir = os.path.join(args.save_dir, str(generation_now), "elites")
        _write_elites(
            elite_dir,
            args,
            elite_batch,
            weight_batch,
            predicted_offspring_objs,
            all_offspring_batch,
        )

        if generation_now == total_generation:
            save_final_ep(args, ep)
            with open(os.path.join(args.save_dir, "elapsed_seconds.txt"), "w") as fp:
                fp.write(f"{time.time() - run_start_time:.3f}\n")

    if getattr(args, "device", "auto") == "auto":
        used_device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    else:
        used_device = args.device
    print(f"Conditioned centralized training finished on learner device: {used_device}")
