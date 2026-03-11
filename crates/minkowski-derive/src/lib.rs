use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derive macro for Table types.
///
/// Generates:
/// - `unsafe impl Bundle` (component registration + put)
/// - `unsafe impl Table` (field count, register)
/// - `FooRef<'w>` and `FooMut<'w>` typed row reference structs
/// - `unsafe impl TableRow` for both Ref and Mut types
#[proc_macro_derive(Table)]
pub fn derive_table(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("#[derive(Table)] only supports structs with named fields"),
        },
        _ => panic!("#[derive(Table)] only supports structs"),
    };

    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
    let field_count = field_names.len();
    let field_indices: Vec<_> = (0..field_count).collect();

    // Generate Bundle impl
    let bundle_impl = quote! {
        unsafe impl ::minkowski::bundle::Bundle for #name {
            fn component_ids(
                registry: &mut ::minkowski::component::ComponentRegistry,
            ) -> Vec<::minkowski::component::ComponentId> {
                let mut ids = vec![
                    #(registry.register::<#field_types>()),*
                ];
                ids.sort_unstable();
                ids.dedup();
                assert_eq!(ids.len(), #field_count, "duplicate component types in table");
                ids
            }

            unsafe fn put(
                self,
                registry: &::minkowski::component::ComponentRegistry,
                func: &mut dyn FnMut(
                    ::minkowski::component::ComponentId,
                    *const u8,
                    std::alloc::Layout,
                ),
            ) {
                #(
                    let #field_names = std::mem::ManuallyDrop::new(self.#field_names);
                    func(
                        registry.id::<#field_types>().unwrap(),
                        &#field_names as *const std::mem::ManuallyDrop<#field_types>
                            as *const #field_types
                            as *const u8,
                        std::alloc::Layout::new::<#field_types>(),
                    );
                )*
            }
        }
    };

    // Ref and Mut type names
    let ref_name = syn::Ident::new(&format!("{}Ref", name), name.span());
    let mut_name = syn::Ident::new(&format!("{}Mut", name), name.span());

    // Generate Table impl
    let table_impl = quote! {
        unsafe impl ::minkowski::table::Table for #name {
            const FIELD_COUNT: usize = #field_count;
            type Ref<'w> = #ref_name<'w>;
            type Mut<'w> = #mut_name<'w>;

            fn register(
                registry: &mut ::minkowski::component::ComponentRegistry,
            ) -> Vec<::minkowski::component::ComponentId> {
                vec![
                    #(registry.register::<#field_types>()),*
                ]
            }
        }
    };

    let ref_struct = quote! {
        pub struct #ref_name<'w> {
            #(pub #field_names: &'w #field_types,)*
        }
    };

    let mut_struct = quote! {
        pub struct #mut_name<'w> {
            #(pub #field_names: &'w mut #field_types,)*
        }
    };

    // Generate TableRow impls
    let ref_row_impl = quote! {
        unsafe impl<'w> ::minkowski::table::TableRow<'w> for #ref_name<'w> {
            unsafe fn from_row(col_ptrs: &[(*mut u8, usize)], row: usize) -> Self {
                Self {
                    #(
                        #field_names: unsafe { &*(col_ptrs[#field_indices].0
                            .add(row * col_ptrs[#field_indices].1)
                            as *const #field_types) },
                    )*
                }
            }
        }
    };

    let mut_row_impl = quote! {
        unsafe impl<'w> ::minkowski::table::TableRow<'w> for #mut_name<'w> {
            unsafe fn from_row(col_ptrs: &[(*mut u8, usize)], row: usize) -> Self {
                Self {
                    #(
                        #field_names: unsafe { &mut *(col_ptrs[#field_indices].0
                            .add(row * col_ptrs[#field_indices].1)
                            as *mut #field_types) },
                    )*
                }
            }
        }
    };

    let expanded = quote! {
        #bundle_impl
        #table_impl
        #ref_struct
        #mut_struct
        #ref_row_impl
        #mut_row_impl
    };

    expanded.into()
}
