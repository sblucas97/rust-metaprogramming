use proc_macro::TokenStream;
use std::{collections::HashMap, fmt::format};
use syn::{parse::Parse, parse_macro_input, Token, ItemFn};

struct Generator {
    file_content: String,
    indent: usize,
}

impl Generator {
    fn new() -> Self {
        Self {
            file_content: String::new(),
            indent: 0,
        }
    }

    fn type_map() -> HashMap<&'static str, &'static str> {
        HashMap::from([
            ("u8", "uint8_t"),
            ("u16", "uint16_t"),
            ("u32", "uint32_t"),
            ("u64", "uint64_t"),
            ("f32", "float"),
            ("f64", "double"),
        ])
    }

    fn indent_str(&self) -> String {
        const INDENT_SIZE: usize = 4;
        " ".repeat(self.indent * INDENT_SIZE)
    }

    fn map_type(&self, t: &str) -> String {
        Self::type_map()
            .get(t)
            .unwrap_or_else(|| panic!("Unsupported Rust type: {}", t))
            .to_string()
    }

    fn map_binop(&self, op: &syn::BinOp) -> &'static str {
        match op {
            syn::BinOp::Add(_) => "+",
            syn::BinOp::Sub(_) => "-",
            syn::BinOp::Mul(_) => "*",
            syn::BinOp::Div(_) => "/",
            syn::BinOp::Rem(_) => "%",
            syn::BinOp::And(_) => "&&",
            syn::BinOp::Or(_) => "||",
            syn::BinOp::BitAnd(_) => "&",
            syn::BinOp::BitOr(_) => "|",
            syn::BinOp::BitXor(_) => "^",
            syn::BinOp::Shl(_) => "<<",
            syn::BinOp::Shr(_) => ">>",
            syn::BinOp::Eq(_) => "==",
            syn::BinOp::Lt(_) => "<",
            syn::BinOp::Le(_) => "<=",
            syn::BinOp::Ne(_) => "!=",
            syn::BinOp::Ge(_) => ">=",
            syn::BinOp::Gt(_) => ">",
            _ => "",
        }
    }

    pub fn gen_output_file(&self, name: &str, extension: &str) {
        // let output_lib_path = format!("./{}.so", name);
        std::fs::write(format!("{}.{}", name, extension), self.file_content.clone())
            .expect("Failed to write kernel file");
    }

    pub fn gen_include_headers_launcher(&mut self, fn_name: &str) {
        let stdio = "#include<stdio.h>";
        let cuda_runtime = "#include<cuda_runtime.h>";
        let include_headers = format!("{}\n{}\n", stdio, cuda_runtime);

        self.file_content.push_str(&include_headers);
        self.file_content.push_str(&format!(r#"#include "generated_{}.h""#, fn_name));
        self.file_content.push_str("\n");
    }

    pub fn gen_header_constants_launcher(&mut self, rows: u64, cols: u64) {
        let define_str = "#define";
        let rows_str = "ROWS";
        let cols_str = "COLS";
        self.file_content
            .push_str(&format!("\n{} {} {}\n", define_str, rows_str, rows));
        self.file_content
            .push_str(&format!("{} {} {}\n\n", define_str, cols_str, cols));
    }

    pub fn extract_param_names(&self, params: &str) -> Vec<String> {
        params
            .split(',')
            .map(|p| {
                let p = p.trim();
                let last = p.split_whitespace().last().unwrap_or("");
                last.trim_start_matches('*')
                    .trim_start_matches('&')
                    .to_string()
            })
            .collect()
    }

    pub fn extract_param_names_joined(&self, params: &str) -> String {
        self.extract_param_names(params).join(", ")
    }

    pub fn gen_launcher_function(&mut self, block_size: u64, fn_name: &str, input_fn: &ItemFn) {
        let params_string = self.gen_kernel_arguments(input_fn);
        self.file_content.push_str(&format!(r#"extern "C" void launch_generated_{}({}) {{"#, fn_name, params_string));
        self.file_content.push_str("\n");
        self.indent += 1;
        self.file_content.push_str(&format!("{}int total = ROWS * COLS; \n", self.indent_str()));
        self.file_content.push_str(&format!("{}int block_size = {};\n", self.indent_str(), block_size));
        self.file_content.push_str(&format!("{}int grid_size = (total + block_size - 1) / block_size;\n", self.indent_str()));
        self.file_content.push_str(&format!("{}{}<<<grid_size, block_size>>>({});\n", self.indent_str(), fn_name, self.extract_param_names_joined(&params_string)));
        self.file_content.push_str(&format!("{}cudaDeviceSynchronize();\n", self.indent_str()));
        self.indent -= 1;
        self.file_content.push_str("}");
    }

    pub fn gen_include_headers(&mut self) {
        self.file_content.push_str("#include<stdio.h>\n");
        self.file_content.push_str("#include<cstdint>\n");
        self.file_content.push_str("#include<cuda_runtime.h>\n\n");
    }

    pub fn gen_kernel_header_headers(&mut self) {
        self.file_content.push_str("#ifndef KERNEL_H\n");
        self.file_content.push_str("#define KERNEL_H\n\n");
        
        self.file_content.push_str("#include<stdio.h>\n");
        self.file_content.push_str("#include<cstdint>\n");
        self.file_content.push_str("#include<cuda_runtime.h>\n\n");
    }

    pub fn gen_kernel_header_end_file(&mut self) {
        self.file_content.push_str("#endif");
    }

    pub fn gen_kernel_header_fn_signature(&mut self, func_name: &str, input_fn: &ItemFn) {
        let fn_signature = self.gen_kernel_signature(func_name, input_fn);
        self.file_content.push_str("\n");
        self.file_content.push_str(&fn_signature);
        self.file_content.push_str(";\n\n");
    }

    pub fn gen_header_constants(&mut self, rows: u64, cols: u64) {
        let define_str = "#define";
        let rows_str = "ROWS";
        let cols_str = "COLS";
        self.file_content
            .push_str(&format!("\n{} {} {}\n", define_str, rows_str, rows));
        self.file_content
            .push_str(&format!("{} {} {}\n\n", define_str, cols_str, cols));
    }

    pub fn gen_kernel_arguments(&mut self, input_fn: &syn::ItemFn) -> String {
        let mut params: Vec<String> = Vec::new();

        for arg in &input_fn.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = arg {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let arg_name = pat_ident.ident.to_string();
                    
                    let (rust_type, is_mut) = match &*pat_type.ty {
                        syn::Type::Reference(type_ref) => {
                            let is_mut = type_ref.mutability.is_some();

                            match &*type_ref.elem {
                                syn::Type::Path(type_path) => {
                                    let segment = type_path.path.segments.last().unwrap();

                                    // Handle CudaVec<T>
                                    if segment.ident == "CudaVec" {
                                        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                                            if let Some(syn::GenericArgument::Type(syn::Type::Path(inner_path))) = args.args.first() {
                                                let inner_segment = inner_path.path.segments.last().unwrap();
                                                (inner_segment.ident.to_string(), is_mut)
                                            } else {
                                                continue;
                                            }
                                        } else {
                                            continue;
                                        }
                                    } else {
                                        (segment.ident.to_string(), is_mut)
                                    }
                                }
                                _ => continue,
                            }
                        }

                        syn::Type::Path(type_path) => {
                            let segment = type_path.path.segments.last().unwrap();

                            if segment.ident == "CudaVec" {
                                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                                    if let Some(syn::GenericArgument::Type(syn::Type::Path(inner_path))) = args.args.first() {
                                        let inner_segment = inner_path.path.segments.last().unwrap();

                                        (inner_segment.ident.to_string(), true)
                                    } else {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            } else {
                                println!("{:?}", type_path);
                                continue;
                            }
                        }
                        _ => continue,
                    };
                    let c_type = self.map_type(&rust_type);
                    let param_repr = if is_mut {
                        format!("{} *{}", c_type, arg_name)
                    } else {
                        format!("const {} *{}", c_type, arg_name)
                    };

                    params.push(param_repr);
                }
            }
        }

        let params_string = params.join(", ");
        params_string
    }

    pub fn gen_kernel_signature(&mut self, fn_name: &str, input_fn: &syn::ItemFn) -> String {
        let params_string = self.gen_kernel_arguments(input_fn);
        format!("__global__ void {}({})", fn_name, params_string)
    }

    pub fn gen_kernel(&mut self, func_name: &str, input_fn: &syn::ItemFn) {
        let fn_signature = self.gen_kernel_signature(func_name, input_fn);
        let mut r = format!("{} {{\n", fn_signature);
        self.indent += 1;

        for stmt in &input_fn.block.stmts {
            r.push_str(&self.gen_stmt(stmt));
        }

        self.indent -= 1;
        r.push_str("}\n");

        self.file_content.push_str(&r);
    }

    pub fn gen_stmt(&mut self, stmt: &syn::Stmt) -> String {
        // println!("{:#?}", stmt);
        match stmt {
            syn::Stmt::Local(local) => {
                let mut s = self.gen_local(local);
                s.push_str(";\n");
                s
            }
            syn::Stmt::Expr(expr, semi) => {
                let mut s = self.gen_expr(expr);
                if semi.is_some() {
                    s.push_str(";\n");
                } else {
                    s.push_str("\n");
                }

                s
            }
            _ => String::new(),
        }
    }

    fn gen_binary(&mut self, expr_binary: &syn::ExprBinary) -> String {
        let left = self.gen_expr(&expr_binary.left);
        let right = self.gen_expr(&expr_binary.right);
        let op = self.map_binop(&expr_binary.op);

        format!("{} {} {}", left, op, right)
    }

    pub fn gen_local(&mut self, local: &syn::Local) -> String {
        println!("{:#?}", local);
        if let syn::Pat::Type(pat_type) = &local.pat {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                let name = pat_ident.ident.to_string();

                if let syn::Type::Path(type_path) = &*pat_type.ty {
                    let type_name = type_path.path.segments.last().unwrap().ident.to_string();

                    let c_type = self.map_type(&type_name);

                    if let Some(init) = &local.init {
                        let value = self.gen_expr(&init.expr);

                        return format!("{}{} {} = {}", self.indent_str(), c_type, name, value);
                    }
                }
            }
        }

        String::new()
    }

    pub fn gen_if(&mut self, expr_if: &syn::ExprIf) -> String {
        let cond = self.gen_expr(&expr_if.cond);

        let mut result = format!("{}if {} {{\n", self.indent_str(), cond);

        self.indent += 1;

        for stmt in &expr_if.then_branch.stmts {
            result.push_str(&self.gen_stmt(stmt));
        }

        self.indent -= 1;

        result.push_str(&format!("{}}}\n", self.indent_str()));

        result
    }

    fn gen_expr(&mut self, expr: &syn::Expr) -> String {
        match expr {
            syn::Expr::If(expr_if) => self.gen_if(expr_if),
            syn::Expr::Lit(expr_lit) => self.gen_lit(expr_lit),
            syn::Expr::Binary(expr_bin) => self.gen_binary(expr_bin),
            syn::Expr::Path(expr_path) => self.gen_path(expr_path),
            syn::Expr::Paren(expr_paren) => self.gen_paren(expr_paren),
            syn::Expr::Field(expr_field) => self.gen_expr_field(expr_field),
            syn::Expr::Assign(expr_assing) => self.gen_expr_assing(expr_assing),
            syn::Expr::Index(expr_index) => self.gen_expr_index(expr_index),
            _ => {
                println!("{:#?}", expr);
                String::new()
            }
        }
    }

    fn gen_expr_index(&mut self, expr_index: &syn::ExprIndex) -> String {
        println!("######################################################");
        println!("######################################################");
        println!("{:?}", expr_index);
        println!("######################################################");
        println!("######################################################");
        let expr = self.gen_expr(&expr_index.expr);
        let idx = self.gen_expr(&expr_index.index);
        format!("{}[{}]", expr, idx)
    }

    fn gen_expr_assing(&mut self, expr_assing: &syn::ExprAssign) -> String {
        let left = self.gen_expr(&expr_assing.left);
        let right = self.gen_expr(&expr_assing.right);
        format!("{}{} = {}", self.indent_str(), left, right)
    }

    fn gen_expr_field(&mut self, expr_field: &syn::ExprField) -> String {
        let base = self.gen_expr(&expr_field.base);
        let member = match &expr_field.member {
            syn::Member::Named(ident) => ident.to_string(),
            syn::Member::Unnamed(_) => panic!("Tuple field access not working yet"),
        };

        format!("{}.{}", base, member)
    }

    fn gen_paren(&mut self, expr_paren: &syn::ExprParen) -> String {
        format!("({})", self.gen_expr(&expr_paren.expr))
    }

    fn gen_path(&mut self, expr_path: &syn::ExprPath) -> String {
        expr_path.path.segments.last().unwrap().ident.to_string()
    }

    fn gen_lit(&self, lit: &syn::ExprLit) -> String {
        match &lit.lit {
            syn::Lit::Int(int_lit) => int_lit.base10_digits().to_string(),
            _ => String::new(),
        }
    }
}

pub fn gen_kernel(attr: &TokenStream, input_fn: ItemFn) {
    let mut kernel_generator = Generator::new();
    kernel_generator.gen_include_headers();
    kernel_generator.gen_header_constants(100, 100);

    let fn_name = input_fn.sig.ident.to_string();
    let name = format!("generated_{fn_name}");
    let extension = "cu";

    kernel_generator.gen_kernel(&fn_name, &input_fn);

    kernel_generator.gen_output_file(&name, &extension);

    let mut kernel_header_generator = Generator::new();
    let h_name = format!("generated_{fn_name}");
    let h_extension = "h";
    kernel_header_generator.gen_kernel_header_headers();
    kernel_header_generator.gen_kernel_header_fn_signature(&fn_name, &input_fn);
    kernel_header_generator.gen_kernel_header_end_file();
    kernel_header_generator.gen_output_file(&h_name, h_extension);
}

pub fn gen_launcher(attr: &TokenStream, input_fn: ItemFn) {
    let mut generator: Generator = Generator::new();
    let fn_name = input_fn.sig.ident.to_string();
    generator.gen_include_headers_launcher(&fn_name);
    generator.gen_header_constants_launcher(100, 100);
    generator.gen_launcher_function(256, &fn_name, &input_fn);
    let extension = "cu";
    let name = format!("generated_{fn_name}_launcher");
    generator.gen_output_file(&name, &extension);
}