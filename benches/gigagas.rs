//! Benchemark mocked blocks that exceed 1 Gigagas.

// TODO: More fancy benchmarks & plots.

#![allow(missing_docs)]

use std::{num::NonZeroUsize, thread};

use alloy_primitives::{Address, U160, U256};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use erc20::contract::ERC20Token;
use pevm::{
    chain::PevmEthereum, execute_revm_sequential, Bytecodes, ChainState, EvmAccount,
    InMemoryStorage, Pevm,
};
use rand::Rng;
use revm::primitives::{BlockEnv, SpecId, TransactTo, TxEnv};
use uniswap::contract::SingleSwap;

// Better project structure
#[path = "../tests/common/mod.rs"]
pub mod common;

#[path = "../tests/erc20/mod.rs"]
pub mod erc20;

#[path = "../tests/uniswap/mod.rs"]
pub mod uniswap;

const GIGA_GAS: u64 = 1_000_000_000;

//#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub fn bench(c: &mut Criterion, name: &str, storage: InMemoryStorage, txs: Vec<TxEnv>) {
    let concurrency_level = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let chain = PevmEthereum::mainnet();
    let spec_id = SpecId::LATEST;
    let block_env = BlockEnv::default();
    let mut pevm = Pevm::default();
    let mut group = c.benchmark_group(name);
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            execute_revm_sequential(
                black_box(&storage),
                black_box(&chain),
                black_box(spec_id),
                black_box(block_env.clone()),
                black_box(txs.clone()),
            )
        })
    });
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let result = pevm.execute_revm_parallel(
                black_box(&storage),
                black_box(&chain),
                black_box(spec_id),
                black_box(block_env.clone()),
                black_box(txs.clone()),
                black_box(concurrency_level),
            );

            assert!(result.is_ok(), "Execution failed: {:?}", result);
        })
    });
    group.finish();
}

pub fn bench_raw_transfers(c: &mut Criterion, db_latency_us: u64) {
    let block_size = (GIGA_GAS as f64 / common::RAW_TRANSFER_GAS_LIMIT as f64).ceil() as usize;
    const START_ADDRESS: usize = 1000;
    const MINER_ADDRESS: usize = 0;
    let mut storage = InMemoryStorage::new(
        std::iter::once(MINER_ADDRESS)
            .chain(START_ADDRESS..START_ADDRESS + block_size)
            .map(common::mock_account),
        None,
        [],
    );
    storage.latency_us = db_latency_us;
    bench(
        c,
        "Independent Raw Transfers",
        storage,
        (0..block_size)
            .map(|i| {
                let address = Address::from(U160::from(START_ADDRESS + i));
                TxEnv {
                    caller: address,
                    transact_to: TransactTo::Call(address),
                    value: U256::from(1),
                    gas_limit: common::RAW_TRANSFER_GAS_LIMIT,
                    gas_price: U256::from(1),
                    ..TxEnv::default()
                }
            })
            .collect::<Vec<_>>(),
    );
}

fn pick_account_idx(num_eoa: usize, hot_ratio: f64) -> usize {
    if hot_ratio <= 0.0 {
        // Uniform workload
        return rand::random::<usize>() % num_eoa;
    }

    // Let `hot_ratio` of transactions conducted by 10% of hot accounts
    let hot_start_idx = (num_eoa as f64 * 0.9) as usize;
    if rand::thread_rng().gen_range(0.0..1.0) < hot_ratio {
        // Access hot
        hot_start_idx + rand::random::<usize>() % (num_eoa - hot_start_idx)
    } else {
        rand::random::<usize>() % hot_start_idx
    }
}

fn bench_dependent_raw_transfers(
    c: &mut Criterion,
    db_latency_us: u64,
    num_eoa: usize,
    hot_ratio: f64,
) {
    let block_size = (GIGA_GAS as f64 / common::RAW_TRANSFER_GAS_LIMIT as f64).ceil() as usize;
    const START_ADDRESS: usize = 1000;
    const MINER_ADDRESS: usize = 0;
    let mut storage = InMemoryStorage::new(
        std::iter::once(MINER_ADDRESS)
            .chain(START_ADDRESS..START_ADDRESS + num_eoa)
            .map(common::mock_account),
        None,
        [],
    );
    storage.latency_us = db_latency_us;

    let mut nonce_vec = vec![1u64; num_eoa];

    bench(
        c,
        "Dependent Raw Transfers",
        storage,
        (0..block_size)
            .map(|_| {
                let from_idx = pick_account_idx(num_eoa, hot_ratio);
                let from = Address::from(U160::from(START_ADDRESS + from_idx));
                let to = Address::from(U160::from(
                    START_ADDRESS + pick_account_idx(num_eoa, hot_ratio),
                ));
                let nonce = nonce_vec[from_idx];
                nonce_vec[from_idx] += 1;
                TxEnv {
                    caller: from,
                    transact_to: TransactTo::Call(to),
                    value: U256::from(1),
                    gas_limit: common::RAW_TRANSFER_GAS_LIMIT,
                    gas_price: U256::from(1),
                    nonce: Some(nonce),
                    ..TxEnv::default()
                }
            })
            .collect::<Vec<_>>(),
    );
}

pub fn bench_erc20(c: &mut Criterion, db_latency_us: u64) {
    let block_size = (GIGA_GAS as f64 / erc20::ESTIMATED_GAS_USED as f64).ceil() as usize;
    let (mut state, bytecodes, txs) = erc20::generate_cluster(block_size, 1, 1);
    state.insert(Address::ZERO, EvmAccount::default()); // Beneficiary
    let mut storage = InMemoryStorage::new(state, Some(&bytecodes), []);
    storage.latency_us = db_latency_us;
    bench(c, "Independent ERC20", storage, txs);
}

fn bench_dependent_erc20(c: &mut Criterion, db_latency_us: u64, num_eoa: usize, hot_ratio: f64) {
    let block_size = (GIGA_GAS as f64 / erc20::ESTIMATED_GAS_USED as f64).ceil() as usize;
    let (mut state, bytecodes, eoa, sca) = erc20::generate_clusters(num_eoa, 1);
    state.insert(Address::ZERO, EvmAccount::default()); // Beneficiary
    let mut txs = Vec::with_capacity(block_size);
    let sca = sca[0];

    let mut nonce_vec = vec![0u64; num_eoa];

    for _ in 0..block_size {
        let from_idx = pick_account_idx(num_eoa, hot_ratio);
        let nonce = nonce_vec[from_idx];
        nonce_vec[from_idx] += 1;
        let from = eoa[from_idx];
        let to = eoa[pick_account_idx(num_eoa, hot_ratio)];
        let tx = TxEnv {
            caller: from,
            transact_to: TransactTo::Call(sca),
            value: U256::from(0),
            gas_limit: erc20::GAS_LIMIT,
            gas_price: U256::from(1),
            data: ERC20Token::transfer(to, U256::from(900)),
            nonce: Some(nonce),
            ..TxEnv::default()
        };
        txs.push(tx);
    }

    let mut db = InMemoryStorage::new(state, Some(&bytecodes), []);
    db.latency_us = db_latency_us;

    bench(c, "Dependent ERC20", db, txs);
}

pub fn bench_uniswap(c: &mut Criterion, db_latency_us: u64) {
    let block_size = (GIGA_GAS as f64 / uniswap::ESTIMATED_GAS_USED as f64).ceil() as usize;
    let mut final_state = ChainState::from_iter([(Address::ZERO, EvmAccount::default())]); // Beneficiary
    let mut final_bytecodes = Bytecodes::default();
    let mut final_txs = Vec::<TxEnv>::new();
    for _ in 0..block_size {
        let (state, bytecodes, txs) = uniswap::generate_cluster(1, 1);
        final_state.extend(state);
        final_bytecodes.extend(bytecodes);
        final_txs.extend(txs);
    }
    let mut storage = InMemoryStorage::new(final_state, Some(&final_bytecodes), []);
    storage.latency_us = db_latency_us;
    bench(c, "Independent Uniswap", storage, final_txs);
}

fn bench_hybrid(c: &mut Criterion, db_latency_us: u64, num_eoa: usize, hot_ratio: f64) {
    // 60% native transfer, 20% erc20 transfer, 20% uniswap
    let num_native_transfer =
        (GIGA_GAS as f64 * 0.6 / common::RAW_TRANSFER_GAS_LIMIT as f64).ceil() as usize;
    let num_erc20_transfer =
        (GIGA_GAS as f64 * 0.2 / erc20::ESTIMATED_GAS_USED as f64).ceil() as usize;
    let num_uniswap = (GIGA_GAS as f64 * 0.2 / uniswap::ESTIMATED_GAS_USED as f64).ceil() as usize;

    const START_ADDRESS: usize = 1000;
    const MINER_ADDRESS: usize = 0;
    let mut state: ChainState = std::iter::once(MINER_ADDRESS)
        .chain(START_ADDRESS..START_ADDRESS + num_eoa)
        .map(common::mock_account)
        .collect();
    let eoa_addresses = state.keys().cloned().collect::<Vec<_>>();
    let mut txs = Vec::with_capacity(num_native_transfer + num_erc20_transfer + num_uniswap);

    let mut nonce_vec = vec![1u64; num_eoa];
    for _ in 0..num_native_transfer {
        let from_idx = pick_account_idx(num_eoa, hot_ratio);
        let nonce = nonce_vec[from_idx];
        nonce_vec[from_idx] += 1;
        let from = Address::from(U160::from(START_ADDRESS + from_idx));
        let to = Address::from(U160::from(
            START_ADDRESS + pick_account_idx(num_eoa, hot_ratio),
        ));
        let tx = TxEnv {
            caller: from,
            transact_to: TransactTo::Call(to),
            value: U256::from(1),
            gas_limit: common::RAW_TRANSFER_GAS_LIMIT,
            gas_price: U256::from(1),
            nonce: Some(nonce),
            ..TxEnv::default()
        };
        txs.push(tx);
    }

    const NUM_ERC20_SCA: usize = 3;
    let (erc20_contract_accounts, erc20_bytecodes) =
        erc20::generate_contract_accounts(NUM_ERC20_SCA, &eoa_addresses);
    for (sca_addr, _) in erc20_contract_accounts.iter() {
        for _ in 0..(num_erc20_transfer / NUM_ERC20_SCA) {
            let from_idx = pick_account_idx(num_eoa, hot_ratio);
            let nonce = nonce_vec[from_idx];
            nonce_vec[from_idx] += 1;
            let from = Address::from(U160::from(START_ADDRESS + from_idx));
            let to = Address::from(U160::from(
                START_ADDRESS + pick_account_idx(num_eoa, hot_ratio),
            ));
            let tx = TxEnv {
                caller: from,
                transact_to: TransactTo::Call(*sca_addr),
                value: U256::from(0),
                gas_limit: erc20::GAS_LIMIT,
                gas_price: U256::from(1),
                data: ERC20Token::transfer(to, U256::from(900)),
                nonce: Some(nonce),
                ..TxEnv::default()
            };
            txs.push(tx);
        }
    }
    state.extend(erc20_contract_accounts.into_iter());

    let mut bytecodes = erc20_bytecodes;
    const NUM_UNISWAP_CLUSTER: usize = 2;
    for _ in 0..NUM_UNISWAP_CLUSTER {
        let (uniswap_contract_accounts, uniswap_bytecodes, single_swap_address) =
            uniswap::generate_contract_accounts(&eoa_addresses);
        state.extend(uniswap_contract_accounts);
        bytecodes.extend(uniswap_bytecodes);
        for _ in 0..(num_uniswap / NUM_UNISWAP_CLUSTER) {
            let data_bytes = if rand::random::<u64>() % 2 == 0 {
                SingleSwap::sell_token0(U256::from(2000))
            } else {
                SingleSwap::sell_token1(U256::from(2000))
            };

            let from_idx = pick_account_idx(num_eoa, hot_ratio);
            let nonce = nonce_vec[from_idx];
            nonce_vec[from_idx] += 1;

            txs.push(TxEnv {
                caller: Address::from(U160::from(START_ADDRESS + from_idx)),
                gas_limit: uniswap::GAS_LIMIT,
                gas_price: U256::from(0xb2d05e07u64),
                transact_to: TransactTo::Call(single_swap_address),
                data: data_bytes,
                nonce: Some(nonce),
                ..TxEnv::default()
            })
        }
    }

    let mut db = InMemoryStorage::new(state, Some(&bytecodes), []);
    db.latency_us = db_latency_us;

    bench(c, "Hybrid", db, txs);
}

pub fn benchmark_gigagas(c: &mut Criterion) {
    let db_latency_us = std::env::var("DB_LATENCY_US")
        .map(|s| s.parse().unwrap())
        .unwrap_or(0);
    let num_eoa = std::env::var("NUM_EOA")
        .map(|s| s.parse().unwrap())
        .unwrap_or(0);
    let hot_ratio = std::env::var("HOT_RATIO")
        .map(|s| s.parse().unwrap())
        .unwrap_or(0.0);

    //bench_raw_transfers(c, db_latency_us);
    bench_dependent_raw_transfers(c, db_latency_us, num_eoa, hot_ratio);
    //bench_erc20(c, db_latency_us);
    bench_dependent_erc20(c, db_latency_us, num_eoa, hot_ratio);
    //bench_uniswap(c, db_latency_us);
    bench_hybrid(c, db_latency_us, num_eoa, hot_ratio);
}

criterion_group!(benches, benchmark_gigagas);
criterion_main!(benches);
