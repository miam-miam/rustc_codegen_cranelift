//! A JIT session that outlives individual rustc sessions. Each build compiles the items a
//! `BuildPolicy` picks into a module of its own, linked to earlier builds through a symbol
//! registry.

use std::ffi::{c_char, c_int};
use std::sync::{Arc, Mutex};

use cranelift_codegen::isa::TargetIsa;
use cranelift_jit::{JITBuilder, JITModule};
use rustc_codegen_ssa::CrateInfo;
use rustc_codegen_ssa::base::{allocator_kind_for_codegen, allocator_shim_contents};
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_hir::def::DefKind;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mono::{MonoItem, MonoItemData};
use rustc_session::config::DebugInfo;
use rustc_span::ErrorGuaranteed;

use super::jit::SymbolLookup;
use super::jit_tls::{self, NewThreadLocal};
use crate::constant::THREAD_LOCAL_ACCESSOR;
use crate::prelude::*;
use crate::unwind_module::UnwindModule;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemAction {
    /// Already defined and unchanged: the latest definition stays.
    Reuse,
    /// Its symbol was never committed.
    Define,
    /// Defined again because its code changed. Code from earlier builds still calls the version
    /// it was compiled with.
    Redefine,
    /// Only its source positions changed. Compiled like `Redefine`, but callers that are reused
    /// keep calling the old copy, with its old line table and panic locations.
    Relocate,
}

/// Decides what each build compiles. Lives across builds, so nothing in it may borrow from one
/// compilation.
pub trait BuildPolicy {
    /// One action per item, in order, before anything is declared. Refusals are emitted through
    /// `tcx.dcx()`, so they read as ordinary compiler errors.
    fn plan<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        items: &[MonoItem<'tcx>],
    ) -> Result<Vec<ItemAction>, ErrorGuaranteed>;

    /// The build finalised. A build that fails after `plan` never gets here, so the policy
    /// records what is defined now rather than in `plan`.
    fn committed(&mut self, batch: &Batch);
}

/// Either way, the session is unchanged.
#[derive(Debug)]
pub enum BuildError {
    /// The policy refused an edit the running program can't take.
    Refused(ErrorGuaranteed),
    /// The build couldn't be compiled.
    Failed(ErrorGuaranteed),
}

#[derive(Debug)]
pub struct Batch {
    pub generation: u32,
    /// Every function in the build, by symbol, with what was done with it.
    pub fns: Vec<(String, ItemAction)>,
}

pub struct JitSession {
    isa: Arc<dyn TargetIsa>,
    dylibs: Arc<SymbolLookup>,
    eh_personality: String,
    /// Register each build's symbols with debuggers through the GDB JIT interface.
    jit_debug_symbols: bool,
    /// Named functions and data defined by committed builds, latest definition winning. Also the
    /// record of what is defined: nothing else in the session tracks it.
    registry: Arc<Mutex<FxHashMap<String, usize>>>,
    /// One per committed build, kept so their code stays mapped.
    modules: Vec<JITModule>,
    /// Successful builds only.
    builds: u32,
    /// Def path of each local `fn()` -> its symbol.
    unit_fns: FxHashMap<String, String>,
    /// Keys given to thread-locals by committed builds, by symbol.
    thread_local_keys: FxHashMap<String, usize>,
}

impl JitSession {
    pub fn new(tcx: TyCtxt<'_>) -> Self {
        let target_cpu = crate::CraneliftCodegenBackend { config: None }.target_cpu(tcx.sess);
        let crate_info = CrateInfo::new(tcx, target_cpu);
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        preload_dylibs_deep_bound(&crate_info);
        JitSession {
            isa: crate::build_isa(tcx.sess, true),
            dylibs: Arc::new(super::jit::dep_symbol_lookup_fn(tcx.sess, crate_info)),
            eh_personality: rustc_symbol_mangling::eh_personality_symbol(tcx).to_string(),
            jit_debug_symbols: tcx.sess.opts.debuginfo != DebugInfo::None,
            registry: Arc::default(),
            modules: Vec::new(),
            builds: 0,
            unit_fns: FxHashMap::default(),
            thread_local_keys: FxHashMap::default(),
        }
    }

    /// Compiles this build's mono items into a fresh module and finalises it. The session only
    /// changes on `Ok`. Errors are emitted through `tcx.dcx()`; fatal errors and panics part-way
    /// through unwind instead of returning, and also drop the module unfinalised.
    pub fn compile(
        &mut self,
        tcx: TyCtxt<'_>,
        policy: &mut dyn BuildPolicy,
    ) -> Result<Batch, BuildError> {
        let cgus = tcx.collect_and_partition_mono_items(()).codegen_units;
        let mono_items = cgus
            .iter()
            .flat_map(|cgu| cgu.items_in_deterministic_order(tcx))
            .collect::<FxIndexMap<_, _>>();
        let items = mono_items.keys().copied().collect::<Vec<_>>();
        let actions = tcx
            .sess
            .time("plan build", || policy.plan(tcx, &items))
            .map_err(BuildError::Refused)?;
        let mut batch = Batch { generation: self.builds + 1, fns: Vec::new() };

        let mut to_define: Vec<(MonoItem<'_>, MonoItemData)> = Vec::new();
        let mut thread_local_images = FxHashSet::default();
        let mut new_unit_fns = Vec::new();
        // Every unsupported item is reported before giving up, not just the first.
        let mut refused = None;
        for ((item, data), action) in mono_items.into_iter().zip(actions) {
            match item {
                MonoItem::Fn(instance) => {
                    let name = tcx.symbol_name(instance).name;
                    batch.fns.push((name.to_owned(), action));
                    if action == ItemAction::Reuse {
                        continue;
                    }
                    if let Some(path) = unit_fn_path(tcx, instance) {
                        new_unit_fns.push((path, name.to_owned()));
                    }
                    if let Err(guar) = refuse_asm(tcx, instance) {
                        refused = Some(guar);
                    }
                    to_define.push((item, data));
                }
                MonoItem::Static(def_id) => {
                    if action == ItemAction::Reuse {
                        continue;
                    }
                    if jit_tls::is_thread_local(tcx, def_id) {
                        thread_local_images.insert(def_id);
                    }
                    to_define.push((item, data));
                }
                MonoItem::GlobalAsm(item_id) => {
                    let item = tcx.hir_item(item_id);
                    refused = Some(
                        tcx.dcx().span_err(item.span, "Global asm is not supported in JIT mode"),
                    );
                }
            }
        }
        if let Some(guar) = refused {
            return Err(BuildError::Failed(guar));
        }

        let first_build = self.modules.is_empty();
        let mut module = self.new_module();
        if first_build && let Some(kind) = allocator_kind_for_codegen(tcx) {
            crate::allocator::codegen(tcx, &mut module, &allocator_shim_contents(tcx, kind));
        }
        super::predefine_mono_items(tcx, &mut module, &to_define);

        let mut debug_context = DebugContext::new(tcx, module.isa(), false, "dummy_cgu_name");
        let mut cached_context = Context::new();
        let output_filenames = tcx.output_filenames(());
        let should_write_ir = crate::pretty_clif::should_write_ir(tcx.sess);
        let committed = &self.thread_local_keys;
        // Thread-locals this build gives keys to. Key = committed count + index.
        let mut new_keys = FxIndexMap::<String, DefId>::default();
        let mut thread_local_key = |symbol: &str, def_id: DefId| match committed.get(symbol) {
            Some(&key) => key,
            None => committed.len() + new_keys.insert_full(symbol.to_owned(), def_id).0,
        };
        tcx.sess.time("codegen mono items", || {
            for &(item, _) in &to_define {
                match item {
                    MonoItem::Fn(instance) => super::jit::codegen_and_compile_fn(
                        tcx,
                        output_filenames,
                        should_write_ir,
                        debug_context.as_mut(),
                        &mut cached_context,
                        &mut module,
                        Some(&mut thread_local_key),
                        instance,
                    ),
                    MonoItem::Static(def_id) if thread_local_images.contains(&def_id) => {
                        crate::constant::codegen_thread_local_image(tcx, &mut module, def_id);
                        // Keyed with its image, even if nothing reads it yet, so any thread-local
                        // keyed by a later build belongs to a dylib.
                        let symbol = tcx.symbol_name(Instance::mono(tcx, def_id)).name;
                        thread_local_key(symbol, def_id);
                    }
                    MonoItem::Static(def_id) => {
                        crate::constant::codegen_static(tcx, &mut module, def_id);
                    }
                    MonoItem::GlobalAsm(_) => unreachable!(),
                }
            }
        });

        // Later builds leave the entry wrapper calling the first build's `main`.
        // The wrapper goes through `lang_start`, which initialises the program's
        // std: its arguments, main thread and `Termination` handling.
        if first_build {
            crate::main_shim::maybe_create_entry_wrapper(tcx, &mut module, false, true);
        }

        let base = self.thread_local_keys.len();
        let new_thread_locals = jit_tls::classify_new_thread_locals(
            tcx,
            new_keys.iter().enumerate().map(|(i, (symbol, &def_id))| (base + i, &**symbol, def_id)),
            &thread_local_images,
            &self.dylibs,
        );

        if let Some(guar) = tcx.dcx().has_errors() {
            return Err(BuildError::Failed(guar));
        }
        let jit = module.finalize_definitions();
        self.commit(jit, new_thread_locals, new_keys.into_keys());
        self.unit_fns.extend(new_unit_fns);
        self.builds = batch.generation;
        policy.committed(&batch);
        Ok(batch)
    }

    fn new_module(&self) -> UnwindModule<JITModule> {
        let mut builder =
            JITBuilder::with_isa(self.isa.clone(), cranelift_module::default_libcall_names());
        crate::compiler_builtins::register_functions_for_jit(&mut builder);
        builder.symbol(THREAD_LOCAL_ACCESSOR, jit_tls::get as *const u8);
        let dylibs = self.dylibs.clone();
        builder.symbol_lookup_fn(Box::new(move |name| (*dylibs)(name)));
        // Lookups are tried newest first, so earlier builds win over the dylibs and `dlsym`.
        let registry = self.registry.clone();
        builder.symbol_lookup_fn(Box::new(move |name| {
            registry.lock().unwrap().get(name).map(|&addr| addr as *const u8)
        }));
        UnwindModule::new(
            JITModule::new(builder),
            &self.eh_personality,
            false,
            self.jit_debug_symbols,
        )
    }

    fn commit(
        &mut self,
        jit: JITModule,
        new_thread_locals: Vec<NewThreadLocal>,
        new_keys: impl Iterator<Item = String>,
    ) {
        let mut registry = self.registry.lock().unwrap();
        let declarations = jit.declarations();
        for (id, decl) in declarations.get_functions() {
            if let Some(name) = &decl.name
                && decl.linkage.is_definable()
            {
                registry.insert(name.clone(), jit.get_finalized_function(id).addr());
            }
        }
        for (id, decl) in declarations.get_data_objects() {
            if let Some(name) = &decl.name
                && decl.linkage.is_definable()
            {
                registry.insert(name.clone(), jit.get_finalized_data(id).0.addr());
            }
        }
        for new in new_thread_locals {
            let (key, source) = jit_tls::source(new, |symbol| registry[symbol], &self.dylibs);
            jit_tls::define(key, source);
        }
        let base = self.thread_local_keys.len();
        self.thread_local_keys.extend(new_keys.enumerate().map(|(i, symbol)| (symbol, base + i)));
        self.modules.push(jit);
    }

    /// The C `main` wrapper defined by the first build.
    pub fn entry(&self) -> Option<extern "C" fn(c_int, *const *const c_char) -> c_int> {
        let addr = *self.registry.lock().unwrap().get("main")?;
        Some(unsafe { std::mem::transmute::<usize, _>(addr) })
    }

    /// The latest version of a non-generic local `fn()`, by def path (`message`, `module::run`).
    pub fn unit_fn(&self, path: &str) -> Option<fn()> {
        let addr = *self.registry.lock().unwrap().get(self.unit_fns.get(path)?)?;
        Some(unsafe { std::mem::transmute::<usize, fn()>(addr) })
    }
}

/// The JIT has no assembler, so asm is refused at its source before any codegen. Codegen would
/// otherwise stop with a fatal error that has no location.
fn refuse_asm<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Result<(), ErrorGuaranteed> {
    if tcx.codegen_instance_attrs(instance.def).flags.contains(CodegenFnAttrFlags::NAKED) {
        let span = tcx.def_span(instance.def_id());
        return Err(tcx.dcx().span_err(span, "Naked functions are not supported in JIT mode"));
    }
    let mut result = Ok(());
    for block in tcx.instance_mir(instance.def).basic_blocks.iter() {
        let terminator = block.terminator();
        if let TerminatorKind::InlineAsm { .. } = terminator.kind {
            let span = terminator.source_info.span;
            result = Err(tcx.dcx().span_err(span, "Inline asm is not supported in JIT mode"));
        }
    }
    result
}

/// The process already contains another copy of std: rustc's own is built into
/// `librustc_driver.so`. glibc binds a library's references to its own exported symbols to the
/// first definition loaded, so a normally loaded `libstd.so` would update the compiler's statics.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn preload_dylibs_deep_bound(crate_info: &CrateInfo) {
    use libloading::os::unix::{Library, RTLD_LAZY, RTLD_LOCAL};
    use rustc_middle::middle::dependency_format::Linkage;
    // glibc's value; libloading doesn't export it.
    const RTLD_DEEPBIND: c_int = 0x8;

    let formats = &crate_info.dependency_formats[&rustc_structures::CrateType::Executable];
    // Dependencies before dependents, as `dep_symbol_lookup_fn` loads them.
    for &cnum in crate_info.used_crates.iter().rev() {
        if formats[cnum] == Linkage::Dynamic
            && let Some(path) = &crate_info.used_crate_source[&cnum].dylib
        {
            let library =
                unsafe { Library::open(Some(path), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND) };
            // Stays loaded for the session, like the libraries `dep_symbol_lookup_fn` leaks.
            std::mem::forget(library.unwrap());
        }
    }
}

fn unit_fn_path(tcx: TyCtxt<'_>, instance: Instance<'_>) -> Option<String> {
    let InstanceKind::Item(def_id) = instance.def else { return None };
    if !def_id.is_local()
        || !instance.args.is_empty()
        || !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn)
    {
        return None;
    }
    let sig = tcx.fn_sig(def_id).instantiate_identity().skip_binder();
    (sig.inputs().is_empty() && sig.output().is_unit()).then(|| tcx.def_path_str(def_id))
}
