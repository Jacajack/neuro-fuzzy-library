#include <iostream>
#include <nlohmann/json.hpp>
#include "nfl_adapter.hpp"

using json = nlohmann::json;

std::unique_ptr<train_context> train_system(const json &j)
{
	const auto &config = j.at("config");
	const auto &dataset = j.at("Xy_train");
	const int num_rules = config.at("num_rules");
	const int clustering_iters = config.at("clustering_iters");
	const int tuning_iters = config.at("tuning_iters");
	const bool normalize = config.at("normalize");
	const double eta = config.at("eta");

	std::cerr << "Will train system..." << std::endl;
	auto ctx = make_system(
		config.at("system_type"),
		config.at("implication"),
		config.at("tnorm"),
		config.at("regression"),
		num_rules,
		clustering_iters,
		tuning_iters,
		normalize,
		eta,
		config.at("minkowski_coeff")
	);

	ctx->train_result = ctx->sys->train(
		dataset,
		num_rules, 
		clustering_iters, 
		tuning_iters, 
		eta, 
		normalize
	);

	return ctx;
}

json test_system(extended_nfs_base& sys, const json &j)
{
	std::cerr << "Will test system..." << std::endl;
	return sys.test(j.at("dataset"));
}

int main()
{
	std::unique_ptr<train_context> ctx;

	while (std::cin.peek() != EOF)
	{
		if (std::cin.peek() == '\n')
		{
			std::cin.get();
			continue;
		}

		json j;
		std::cin >> j;

		if (j.at("cmd") == "train")
		{
			ctx = train_system(j);
			std::cout << ctx->train_result << std::endl;
		}
		else if (j.at("cmd") == "test")
		{
			if (!ctx || !ctx->sys)
				throw std::runtime_error{"not trained"};
			
			json test_result = test_system(*ctx->sys, j);
			std::cout << test_result << std::endl;
		}
		else
			throw std::runtime_error{"invalid cmd"};
	}

	return 0;
}
