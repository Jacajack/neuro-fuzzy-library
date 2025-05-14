#include "nfl_adapter.hpp"
#include <stdexcept>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#include <tnorms/t-norm-product.h>
#include <implications/imp-fodor.h>
#include <implications/imp-goedel.h>
#include <implications/imp-goguen.h>
#include <implications/imp-kleene-dienes.h>
#include <implications/imp-lukasiewicz.h>
#include <implications/imp-reichenbach.h>
#include <implications/imp-rescher.h>
#include <implications/imp-zadeh.h>
#include <tnorms/t-norm.h>
#include <tnorms/t-norm-dombi.h>
#include <tnorms/t-norm-drastic.h>
#include <tnorms/t-norm-dubois-prade.h>
#include <tnorms/t-norm-einstein.h>
#include <tnorms/t-norm-fodor.h>
#include <tnorms/t-norm-frank.h>
#include <tnorms/t-norm-hamacher.h>
#include <tnorms/t-norm-lukasiewicz.h>
#include <tnorms/t-norm-min.h>
#include <tnorms/t-norm-parametrized.h>
#include <tnorms/t-norm-product.h>
#include <tnorms/t-norm-schweizer-sklar.h>
#include <tnorms/t-norm-sugeno-weber.h>
#include <tnorms/t-norm-yager.h>
#include <neuro-fuzzy/annbfis.h>
#include <neuro-fuzzy/annbfis_prototype.h>
#include <neuro-fuzzy/fac_prototype_minkowski_classification.h>
#include <neuro-fuzzy/fac_prototype_minkowski_regression.h>
#include <common/number.h>
#include <partitions/fcm.h>
#pragma GCC diagnostic pop
#pragma clang diagnostic pop


extended_annbfis::extended_annbfis(
	int nRules,
	int nClusteringIterations,
	int nTuningIterations,
	double dbLearningCoefficient,
	bool bNormalisation,
	const ksi::t_norm& tnorm,
	const ksi::implication& imp) :
	ksi::abstract_annbfis{
		nRules,
		nClusteringIterations,
		nTuningIterations,
		dbLearningCoefficient,
		bNormalisation,
		tnorm,
		imp,
		ksi::fcm(nRules, nClusteringIterations)
	},
	extended_nfs<ksi::annbfis>{
		nRules,
		nClusteringIterations,
		nTuningIterations,
		dbLearningCoefficient,
		bNormalisation,
		tnorm,
		imp,
	}
{
}

extended_annbfis::extended_annbfis(
	int nRules,
	int nClusteringIterations,
	int nTuningIterations,
	double dbLearningCoefficient,
	bool bNormalisation,
	const ksi::t_norm& tnorm,
	const ksi::implication& imp,
	double positive_class,
	double negative_class,
	ksi::roc_threshold th) :
	ksi::abstract_annbfis{
		nRules,
		nClusteringIterations,
		nTuningIterations,
		dbLearningCoefficient,
		bNormalisation,
		tnorm,
		imp,
		ksi::fcm(nRules, nClusteringIterations),
		positive_class,
		negative_class,
		th
	},
	extended_nfs<ksi::annbfis>{
		nRules,
		nClusteringIterations,
		nTuningIterations,
		dbLearningCoefficient,
		bNormalisation,
		tnorm,
		imp,
		positive_class,
		negative_class,
		th
	}
{
}


ksi::dataset load_json_dataset(const json &j)
{
	ksi::dataset ds;
	int item_number = 0;

	for (const auto &datum : j)
	{
		ksi::datum d;
		for (const auto &value : datum)
			d.push_back(ksi::number{value.get<double>()});
		d.setID(item_number++);
		d.setIDincomplete(-1);
		ds.addDatum(d);	
	}

	return ds;
}


std::unique_ptr<ksi::implication> make_implication(const std::string &name)
{
	#define MATCH(val, type) do { if (name == val) return std::make_unique<type>(); } while (false)
	MATCH("fodor", ksi::imp_fodor);
	MATCH("goedel", ksi::imp_goedel);
	MATCH("goguen", ksi::imp_goguen);
	MATCH("kleene_dienes", ksi::imp_kleene_dienes);
	MATCH("lukasiewicz", ksi::imp_lukasiewicz);
	MATCH("reichenbach", ksi::imp_reichenbach);
	MATCH("rescher", ksi::imp_rescher);
	MATCH("zadeh", ksi::imp_zadeh);
	#undef MATCH
	throw std::runtime_error{"invalid implication name"};
}

std::unique_ptr<ksi::t_norm> make_tnorm(const std::string &name)
{
	#define MATCH(val, type) do { if (name == val) return std::make_unique<type>(); } while (false)
    MATCH("drastic", ksi::t_norm_drastic);
    MATCH("einstein", ksi::t_norm_einstein);
    MATCH("fodor", ksi::t_norm_fodor);
    MATCH("lukasiewicz", ksi::t_norm_lukasiewicz);
    MATCH("min", ksi::t_norm_min);
    MATCH("product", ksi::t_norm_product);
	#undef MATCH
	throw std::runtime_error{"invalid tnorm name"};
}

std::unique_ptr<train_context> make_system(
	const std::string &system_type,
	const std::string &implication,
	const std::string &tnorm,
	bool regression_task,
	int num_rules,
	int clustering_iters,
	int tuning_iters,
	bool normalize,
	double eta,
	double minkowski_coeff)
{
	auto ctx = std::make_unique<train_context>();
	const ksi::roc_threshold th{ksi::roc_threshold::mean};
	const double pos_class = 1;
	const double neg_class = 1;

	ctx->impl = make_implication(implication);
	ctx->tnorm = make_tnorm(tnorm);

	if (system_type == "annbfis" && regression_task)
	{
		ctx->sys = std::make_unique<extended_annbfis>(
			num_rules,
			clustering_iters,
			tuning_iters,
			eta,
			normalize,
			*ctx->tnorm,
			*ctx->impl
		);
	}
	else if (system_type == "annbfis" && !regression_task)
	{
		ctx->sys = std::make_unique<extended_annbfis>(
			num_rules,
			clustering_iters,
			tuning_iters,
			eta,
			normalize,
			*ctx->tnorm,
			*ctx->impl,
			pos_class,
			neg_class,
			th
		);
	}
	else if (system_type == "annbfis_prototype" && regression_task)
	{
		ctx->fac_proto = std::make_unique<ksi::fac_prototype_minkowski_regression>(minkowski_coeff);
		ctx->sys = std::make_unique<extended_annbfis_prototype>(
			num_rules,
			clustering_iters,
			tuning_iters,
			eta,
			normalize,
			*ctx->impl,
			*ctx->fac_proto
		);
	}
	else if (system_type == "annbfis_prototype" && !regression_task)	
	{
		ctx->fac_proto = std::make_unique<ksi::fac_prototype_minkowski_classification>(minkowski_coeff, 1, 0);
		ctx->sys = std::make_unique<extended_annbfis_prototype>(
			num_rules,
			clustering_iters,
			tuning_iters,
			eta,
			normalize,
			*ctx->impl,
			*ctx->fac_proto,
			pos_class,
			neg_class,
			th
		);
	}
	else if (system_type == "tsk_prototype" && regression_task)
	{
		ctx->fac_proto = std::make_unique<ksi::fac_prototype_minkowski_regression>(minkowski_coeff);
		ctx->sys = std::make_unique<extended_tsk_prototype>(
			num_rules,
			clustering_iters,
			tuning_iters,
			eta,
			normalize,
			*ctx->fac_proto
		);
	}
	else if (system_type == "tsk_prototype" && !regression_task)	
	{
		ctx->fac_proto = std::make_unique<ksi::fac_prototype_minkowski_classification>(minkowski_coeff, 1, 0);
		ctx->sys = std::make_unique<extended_tsk_prototype>(
			num_rules,
			clustering_iters,
			tuning_iters,
			eta,
			normalize,
			*ctx->fac_proto,
			pos_class,
			neg_class,
			th
		);
	}
	else
		throw std::runtime_error{"bad system_type"};

	return ctx;
}