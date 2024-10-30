pub mod contract;

use contract::ERC20Token;
use pevm::{Bytecodes, ChainState, EvmAccount};
use revm::primitives::{uint, Address, TransactTo, TxEnv, U256};

pub const GAS_LIMIT: u64 = 35_000;
pub const ESTIMATED_GAS_USED: u64 = 29_738;

// TODO: Better randomness control. Sometimes we want duplicates to test
// dependent transactions, sometimes we want to guarantee non-duplicates
// for independent benchmarks.
fn generate_addresses(length: usize) -> Vec<Address> {
    (0..length).map(|_| Address::new(rand::random())).collect()
}

/// Return a tuple of (state, bytecodes, eoa_addresses, sca_addresses)
pub(crate) fn generate_clusters(
    num_eoa: usize,
    num_sca: usize,
) -> (ChainState, Bytecodes, Vec<Address>, Vec<Address>) {
    let mut state = ChainState::default();
    let eoa_addresses: Vec<Address> = generate_addresses(num_eoa);

    for person in eoa_addresses.iter() {
        state.insert(
            *person,
            EvmAccount {
                balance: uint!(4_567_000_000_000_000_000_000_U256),
                ..EvmAccount::default()
            },
        );
    }

    let (contract_accounts, bytecodes) = generate_contract_accounts(num_sca, &eoa_addresses);
    let mut erc20_sca_addresses = Vec::with_capacity(num_sca);
    for (addr, sca) in contract_accounts {
        state.insert(addr, sca);
        erc20_sca_addresses.push(addr);
    }

    (state, bytecodes, eoa_addresses, erc20_sca_addresses)
}

/// Return a tuple of (contract_accounts, bytecodes)
pub(crate) fn generate_contract_accounts(
    num_sca: usize,
    eoa_addresses: &[Address],
) -> (Vec<(Address, EvmAccount)>, Bytecodes) {
    let mut accounts = Vec::with_capacity(num_sca);
    let mut bytecodes = Bytecodes::default();
    for _ in 0..num_sca {
        let gld_address = Address::new(rand::random());
        let mut gld_account =
            ERC20Token::new("Gold Token", "GLD", 18, 222_222_000_000_000_000_000_000u128)
                .add_balances(&eoa_addresses, uint!(1_000_000_000_000_000_000_U256))
                .build();
        bytecodes.insert(
            gld_account.code_hash.unwrap(),
            gld_account.code.take().unwrap(),
        );
        accounts.push((gld_address, gld_account));
    }
    (accounts, bytecodes)
}

pub fn generate_cluster(
    num_families: usize,
    num_people_per_family: usize,
    num_transfers_per_person: usize,
) -> (ChainState, Bytecodes, Vec<TxEnv>) {
    let families: Vec<Vec<Address>> = (0..num_families)
        .map(|_| generate_addresses(num_people_per_family))
        .collect();

    let people_addresses: Vec<Address> = families.clone().into_iter().flatten().collect();

    let gld_address = Address::new(rand::random());

    let gld_account = ERC20Token::new("Gold Token", "GLD", 18, 222_222_000_000_000_000_000_000u128)
        .add_balances(&people_addresses, uint!(1_000_000_000_000_000_000_U256))
        .build();

    let mut state = ChainState::from_iter([(gld_address, gld_account)]);
    let mut txs = Vec::new();

    for person in people_addresses.iter() {
        state.insert(
            *person,
            EvmAccount {
                balance: uint!(4_567_000_000_000_000_000_000_U256),
                ..EvmAccount::default()
            },
        );
    }

    for nonce in 0..num_transfers_per_person {
        for family in families.iter() {
            for person in family {
                let recipient = family[(rand::random::<usize>()) % (family.len())];
                let calldata = ERC20Token::transfer(recipient, U256::from(rand::random::<u8>()));

                txs.push(TxEnv {
                    caller: *person,
                    gas_limit: GAS_LIMIT,
                    gas_price: U256::from(0xb2d05e07u64),
                    transact_to: TransactTo::Call(gld_address),
                    data: calldata,
                    nonce: Some(nonce as u64),
                    ..TxEnv::default()
                })
            }
        }
    }

    let mut bytecodes = Bytecodes::default();
    for account in state.values_mut() {
        let code = account.code.take();
        if let Some(code) = code {
            bytecodes.insert(account.code_hash.unwrap(), code);
        }
    }

    (state, bytecodes, txs)
}
