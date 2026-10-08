//! Isolated post-Sin fusion. Original ONNX Sin and Pow kernels are retained.
use ort::{operator::{Operator, OperatorDomain,
    io::{OperatorInput, OperatorOutput}, kernel::{Kernel, KernelAttributes, KernelContext}},
    tensor::TensorElementType};
struct PostSnake;
impl Operator for PostSnake {
    fn name(&self)->&str { "PostSnake" }
    fn inputs(&self)->Vec<OperatorInput> { (0..3).map(|_|OperatorInput::required(TensorElementType::Float32)).collect() }
    fn outputs(&self)->Vec<OperatorOutput> { vec![OperatorOutput::required(TensorElementType::Float32)] }
    fn create_kernel(&self,_:&KernelAttributes)->ort::Result<Box<dyn Kernel>> {
        Ok(Box::new(|ctx:&KernelContext| {
            let x=ctx.input(0)?.ok_or_else(||ort::Error::new("missing x"))?;
            let p=ctx.input(1)?.ok_or_else(||ort::Error::new("missing original Pow"))?;
            let b=ctx.input(2)?.ok_or_else(||ort::Error::new("missing scale"))?;
            let (shape,x)=x.try_extract_tensor::<f32>()?;
            let (p_shape,p)=p.try_extract_tensor::<f32>()?;
            let (b_shape,b)=b.try_extract_tensor::<f32>()?;
            if shape.len()!=3 || shape!=p_shape || b_shape.len()!=3 || b_shape[0]!=1 || b_shape[1]!=shape[1] || b_shape[2]!=1 {
                return Err(ort::Error::new("Unexpected post-Snake broadcast; refuse experiment"));
            }
            let mut z=ctx.output(0,shape.to_vec())?.ok_or_else(||ort::Error::new("missing output"))?;
            let (_,out)=z.try_extract_tensor_mut::<f32>()?;
            let width=shape[2] as usize;
            let channels=shape[1] as usize;
            if std::env::var_os("QWEN_DIAG_ACTIVATION_PARALLEL").is_some() {
                let address=out.as_mut_ptr() as usize;
                // ORT joins its own pool before returning; each row has disjoint writes.
                ctx.par_for(x.len()/width, 0, |row| {
                    let scale=b[row%channels];
                    let base=row*width;
                    let out_row=unsafe { std::slice::from_raw_parts_mut((address as *mut f32).add(base),width) };
                    for i in 0..width {
                        let product=p[base+i]*scale;
                        out_row[i]=x[base+i]+product;
                    }
                })?;
            } else {
                for (row,(x_row,p_row)) in x.chunks_exact(width).zip(p.chunks_exact(width)).enumerate() {
                    let scale=b[row%channels];
                    let out_row=&mut out[row*width..(row+1)*width];
                    for i in 0..width {
                        let product=p_row[i]*scale;
                        out_row[i]=x_row[i]+product;
                    }
                }
            }
            Ok(())
        }))
    }
}
pub(crate) fn domain()->ort::Result<OperatorDomain> { OperatorDomain::new("qwen.diag")?.add(PostSnake) }
