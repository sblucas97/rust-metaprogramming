use proc_macro::TokenStream;
use std::{collections::HashMap};
use syn::{ItemFn};

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

    pub fn gen_include_headers(&mut self) {
        self.file_content.push_str("#include<stdio.h>\n");
        self.file_content.push_str("#include<cstdint>\n");
        self.file_content.push_str("#include<cuda_runtime.h>\n\n");
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
                    
                    let (rust_type, is_mut, is_ref) = match &*pat_type.ty {
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
                                                (inner_segment.ident.to_string(), is_mut, true)
                                            } else {
                                                continue;
                                            }
                                        } else {
                                            continue;
                                        }
                                    } else {
                                        (segment.ident.to_string(), is_mut, true)
                                    }
                                }
                                _ => {
                                    println!("{:#?}", type_ref);
                                    continue
                                },
                            }
                        }

                        syn::Type::Path(type_path) => {
                            let segment = type_path.path.segments.last().unwrap();

                            if segment.ident == "CudaVec" {
                                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                                    if let Some(syn::GenericArgument::Type(syn::Type::Path(inner_path))) = args.args.first() {
                                        let inner_segment = inner_path.path.segments.last().unwrap();

                                        (inner_segment.ident.to_string(), true, false)
                                    } else {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            } else {
                                (segment.ident.to_string(), false, false)
                            }
                        }
                        _ => {
                            continue
                        },
                    };
                    
                    let c_type = self.map_type(&rust_type);
                    let param_repr = match (is_ref, is_mut) {
                        (false, _)     => format!("{} {}", c_type, arg_name),          // plain value: int x
                        (true, true)   => format!("{} *{}", c_type, arg_name),         // mut ref:     int *x
                        (true, false)  => format!("const {} *{}", c_type, arg_name),   // shared ref:  const int *x
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
        format!("extern \"C\" __global__ void {}({})", fn_name, params_string)
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
    
    pub fn gen_device_fn_signature(&mut self, fn_name: &str, input_fn: &syn::ItemFn) -> String {
        let return_type = match &input_fn.sig.output {
            syn::ReturnType::Default => "()".to_string(),
            syn::ReturnType::Type(_, ty) => {
                match ty.as_ref() {
                    syn::Type::Path(type_path) => {
                        // e.g. -> MyStruct, -> Result<T>, -> u32
                        let ty_name = type_path.path.segments
                            .last()
                            .unwrap()
                            .ident.to_string();
                        self.map_type(&ty_name)
                    }
                    syn::Type::Reference(type_ref) => {
                        // e.g. -> &str, -> &MyStruct
                        format!("&{:?}", type_ref.elem)
                    }
                    syn::Type::Tuple(tuple) if tuple.elems.is_empty() => {
                        // explicit -> ()
                        "()".to_string()
                    }
                    _ => {
                        println!("Not recognized return type in gen_device_fn_signature");
                        "()".to_string()
                    }
                }
            },
        };

        let params_string = self.gen_kernel_arguments(input_fn);
        format!("__device__ {} {}({})", return_type, fn_name, params_string)
    }

    fn gen_device_functions(&mut self, device_fns: HashMap<String, syn::ItemFn>) {
        // println!("######################################################");
        // println!("{:#?}", device_fns);
        // println!("######################################################");

        for (fn_name, item_fn) in device_fns.iter() {
            let fn_signature = self.gen_device_fn_signature(fn_name, item_fn);
            let mut r = format!("{} {{\n", fn_signature);
            self.indent += 1;

            for stmt in &item_fn.block.stmts {
                r.push_str(&self.gen_stmt(stmt));
            }

            self.indent -= 1;
            r.push_str("}\n\n");

            self.file_content.push_str(&r);
        }

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

    fn map_unop(&self, op: &syn::UnOp) -> &'static str {
        match op {
            syn::UnOp::Neg(_) => "-",
            syn::UnOp::Not(_) => "!",
            syn::UnOp::Deref(_) => "*",
            _ => panic!("Unsupported unary operator in kernel codegen"),
        }
    }

    fn gen_unary(&mut self, expr_unary: &syn::ExprUnary) -> String {
        let op = self.map_unop(&expr_unary.op);
        let operand = self.gen_expr(&expr_unary.expr);
        format!("{}{}", op, operand)
    }

    fn gen_binary(&mut self, expr_binary: &syn::ExprBinary) -> String {
        let left = self.gen_expr(&expr_binary.left);
        let right = self.gen_expr(&expr_binary.right);
        let op = self.map_binop(&expr_binary.op);

        format!("{} {} {}", left, op, right)
    }

    pub fn gen_local(&mut self, local: &syn::Local) -> String {
        // println!("{:#?}", local);
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

        // C requires parentheses around the condition.
        let mut result = format!("{}if ({}) {{\n", self.indent_str(), cond);

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
            syn::Expr::Call(expr_call) => self.gen_expr_call(expr_call),
            syn::Expr::Return(expr_return) => self.gen_expr_return(expr_return),
            syn::Expr::ForLoop(expr_for_loop) => self.gen_expr_for_loop(expr_for_loop),
            syn::Expr::Cast(expr_cast) => self.gen_expr_cast(expr_cast),
            syn::Expr::Unary(expr_unary) => self.gen_unary(expr_unary),
            _ => {
                println!("{:#?}", expr);
                String::new()
            }
        }
    }

    fn gen_expr_cast(&mut self, expr_cast: &syn::ExprCast) -> String {
        let inner = self.gen_expr(&expr_cast.expr);
        match &*expr_cast.ty {
            syn::Type::Path(type_path) => {
                let ty = type_path.path.segments.last().unwrap().ident.to_string();
                let c = self.map_type(&ty);
                format!("(({})({}))", c, inner)
            }
            _ => panic!("Unsupported cast target type in kernel codegen"),
        }
    }

    fn gen_expr_for_loop(&mut self, expr_for_loop: &syn::ExprForLoop) -> String{
        
        let loop_var = match &*expr_for_loop.pat {
            syn::Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
            _ => panic!("Unsupported for loop pattern"),
        };

        let (start, end, step) = match &*expr_for_loop.expr {
            syn::Expr::MethodCall(method_call) if method_call.method == "step_by" => {
                let step = self.gen_expr(&method_call.args[0]);

                // Tuple receiver: (idx, n)
                match &*method_call.receiver {
                    syn::Expr::Tuple(tuple) if tuple.elems.len() == 2 => {
                        let start = self.gen_expr(&tuple.elems[0]);
                        let end = self.gen_expr(&tuple.elems[1]);
                        (start, end, step)
                    }
                    _ => panic!("Expected a 2-element tuple as step_by receiver"),
                }
            }
            _ => panic!("Unsupported for loop iterator expression"),
        };

        self.indent += 1;
        let mut body_stmts = String::new();
        for stmt in &expr_for_loop.body.stmts {
            body_stmts.push_str(&self.gen_stmt(stmt));
        }
        self.indent -= 1;

        format!(
            "{indent_beginning}for (int {var} = {start}; {var} < {end}; {var} += {step}) {{\n{body}{indent_end}}}",
            indent_beginning = self.indent_str(),
            var = loop_var,
            start = start,
            end = end,
            step = step,
            body = body_stmts,
            indent_end = self.indent_str()
        )
    }

    fn gen_expr_return(&mut self, expr_return: &syn::ExprReturn) -> String {
        let return_val = match &expr_return.expr {
            Some(expr) => self.gen_expr(expr),
            None => panic!("Return value needed")
        };
        format!("{}return {}", self.indent_str(), return_val)
    }

    fn gen_expr_call(&mut self, expr_call: &syn::ExprCall) -> String {
        let func = self.gen_expr(&expr_call.func);

        let args = match &expr_call.args.first() {
            Some(expr) => self.gen_expr(expr),
            None => String::new()
        };
        format!("{}({})", func, args)
    }

    fn gen_expr_index(&mut self, expr_index: &syn::ExprIndex) -> String {
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
            syn::Lit::Float(float_lit) => float_lit.base10_digits().to_string(),
            l => panic!("gen_lit error: {:?} was not implemented yet", l),
        }
    }

}

pub fn gen_kernel(
    _attr: &TokenStream, 
    input_fn: ItemFn, 
    device_fns: HashMap<String, syn::ItemFn>,
    rows: Option<u64>,
    cols: Option<u64>
) {
    let mut kernel_generator = Generator::new();
    kernel_generator.gen_include_headers();

    match (rows, cols) {
        (None, None) => {},
        (Some(_), None) => {},
        (None, Some(_)) => {},
        (Some(r), Some(c)) => kernel_generator.gen_header_constants(r, c),
    }

    let fn_name = input_fn.sig.ident.to_string();
    let name = format!("generated_{fn_name}");
    let extension = "cu";

    kernel_generator.gen_device_functions(device_fns);
    kernel_generator.gen_kernel(&fn_name, &input_fn);

    kernel_generator.gen_output_file(&name, &extension);
}
