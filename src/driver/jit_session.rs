//! A JIT session that outlives individual rustc sessions. Each build compiles every function from
//! the local crate, plus any code from other crates not yet defined into a module of its own,
//! linked to earlier builds through a symbol registry.

use std::ffi::{c_char, c_int};
use std::sync::{Arc, Mutex};

use cranelift_codegen::isa::TargetIsa;
use cranelift_jit::{JITBuilder, JITModule};
use rustc_codegen_ssa::CrateInfo;
use rustc_codegen_ssa::base::{allocator_kind_for_codegen, allocator_shim_contents};
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_hir::def::DefKind;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mono::{MonoItem, MonoItemData};
use rustc_span::ErrorGuaranteed;

use crate::prelude::*;
use crate::unwind_module::UnwindModule;

#[derive(Debug, Default)]
pub struct BuildReport {
    pub build: u32,
    pub new_fns: usize,
    /// Local functions defined by an earlier build and compiled again. Only this and later builds
    /// call the new versions; code from earlier builds still calls the version it was compiled with.
    pub recompiled_fns: usize,
    pub reused_fns: usize,
}

type SymbolLookup = Box<dyn Fn(&str) -> Option<*const u8> + Send>;

pub struct JitSession {
    isa: Arc<dyn TargetIsa>,
    dylibs: Arc<Mutex<SymbolLookup>>,
    eh_personality: String,
    /// Named functions and data defined by committed builds, latest definition winning. Also the
    /// record of what is defined: nothing else in the session tracks it.
    registry: Arc<Mutex<FxHashMap<String, usize>>>,
    /// One per committed build, kept so their code stays mapped.
    modules: Vec<JITModule>,
    /// Successful builds only.
    builds: u32,
    /// Def path of each local `fn()` -> its symbol.
    unit_fns: FxHashMap<String, String>,
}

impl JitSession {
    pub fn new(tcx: TyCtxt<'_>) -> Self {
        let target_cpu = crate::CraneliftCodegenBackend { config: None }.target_cpu(tcx.sess);
        let crate_info = CrateInfo::new(tcx, target_cpu);
        JitSession {
            isa: crate::build_isa(tcx.sess, true),
            dylibs: Arc::new(Mutex::new(super::jit::dep_symbol_lookup_fn(tcx.sess, crate_info))),
            eh_personality: rustc_symbol_mangling::eh_personality_symbol(tcx).to_string(),
            registry: Arc::default(),
            modules: Vec::new(),
            builds: 0,
            unit_fns: FxHashMap::default(),
        }
    }

    /// Compiles this build's mono items into a fresh module and finalises it. The session only
    /// changes on `Ok`. Errors are emitted through `tcx.dcx()`; fatal errors and panics part-way
    /// through unwind instead of returning, and also drop the module unfinalised.
    pub fn compile(&mut self, tcx: TyCtxt<'_>) -> Result<BuildReport, ErrorGuaranteed> {
        let mut report = BuildReport { build: self.builds + 1, ..BuildReport::default() };

        let cgus = tcx.collect_and_partition_mono_items(()).codegen_units;
        let mono_items = cgus
            .iter()
            .flat_map(|cgu| cgu.items_in_deterministic_order(tcx))
            .collect::<FxHashMap<_, _>>();

        let mut to_define: Vec<(MonoItem<'_>, MonoItemData)> = Vec::new();
        let mut new_unit_fns = Vec::new();
        // Every unsupported item is reported before giving up, not just the first.
        let mut refused = None;
        for (item, data) in mono_items {
            match item {
                MonoItem::Fn(instance) => {
                    let name = tcx.symbol_name(instance).name;
                    let defined = self.registry.lock().unwrap().contains_key(name);
                    // Other crates' code can't change during a session.
                    if defined && !instance.def_id().is_local() {
                        report.reused_fns += 1;
                        continue;
                    }
                    if defined {
                        report.recompiled_fns += 1;
                    } else {
                        report.new_fns += 1;
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
                    // Statics keep their storage, so a changed initialiser has no effect.
                    let name = tcx.symbol_name(Instance::mono(tcx, def_id)).name;
                    if !self.registry.lock().unwrap().contains_key(name) {
                        to_define.push((item, data));
                    }
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
            return Err(guar);
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
                        None,
                        instance,
                    ),
                    MonoItem::Static(def_id) => {
                        crate::constant::codegen_static(tcx, &mut module, def_id);
                    }
                    MonoItem::GlobalAsm(_) => unreachable!(),
                }
            }
        });

        // Later builds leave the entry wrapper calling the first build's `main`.
        if first_build {
            crate::main_shim::maybe_create_entry_wrapper(tcx, &mut module, true, true);
        }

        if let Some(guar) = tcx.dcx().has_errors() {
            return Err(guar);
        }
        let jit = module.finalize_definitions();
        self.commit(jit);
        self.unit_fns.extend(new_unit_fns);
        self.builds = report.build;
        Ok(report)
    }

    fn new_module(&self) -> UnwindModule<JITModule> {
        let mut builder =
            JITBuilder::with_isa(self.isa.clone(), cranelift_module::default_libcall_names());
        crate::compiler_builtins::register_functions_for_jit(&mut builder);
        let dylibs = self.dylibs.clone();
        builder.symbol_lookup_fn(Box::new(move |name| (**dylibs.lock().unwrap())(name)));
        // Lookups are tried newest first, so earlier builds win over the dylibs and `dlsym`.
        let registry = self.registry.clone();
        builder.symbol_lookup_fn(Box::new(move |name| {
            registry.lock().unwrap().get(name).map(|&addr| addr as *const u8)
        }));
        UnwindModule::new(JITModule::new(builder), &self.eh_personality, false, false)
    }

    fn commit(&mut self, jit: JITModule) {
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
