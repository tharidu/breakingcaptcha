package kth.id2223;

/**
 * Created by tharidu on 12/10/16.
 */

import nl.captcha.text.renderer.WordRenderer;

import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.font.GlyphVector;
import java.awt.font.TextAttribute;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.security.SecureRandom;
import java.text.AttributedCharacterIterator;
import java.text.AttributedString;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Renders the answer onto the image.
 *
 * @author <a href="mailto:james.childers@gmail.com">James Childers</a>
 */
public class DefaultWordRenderer1 implements WordRenderer {

    private static final Random RAND = new SecureRandom();
    private static final List<Color> DEFAULT_COLORS = new ArrayList<Color>();
    private static final List<Font> DEFAULT_FONTS = new ArrayList<Font>();
    // The text will be rendered 25%/5% of the image height/width from the X and Y axes
    private static final double YOFFSET = 0.25;
    private static final double XOFFSET = 0.05;

    static {
        DEFAULT_COLORS.add(Color.BLACK);
        DEFAULT_FONTS.add(new Font("Arial", Font.BOLD, 50));
        DEFAULT_FONTS.add(new Font("Courier", Font.BOLD, 50));
    }

    private final List<Color> _colors = new ArrayList<Color>();
    private final List<Font> _fonts = new ArrayList<Font>();

    /**
     * Use the default color (black) and fonts (Arial and Courier).
     */
    public DefaultWordRenderer1() {
        this(DEFAULT_COLORS, DEFAULT_FONTS);
    }

    /**
     * Build a <code>WordRenderer</code> using the given <code>Color</code>s and
     * <code>Font</code>s.
     *
     * @param colors
     * @param fonts
     */
    public DefaultWordRenderer1(List<Color> colors, List<Font> fonts) {
        _colors.addAll(colors);
        _fonts.addAll(fonts);
    }

    /**
     * Render a word onto a BufferedImage.
     *
     * @param word  The word to be rendered.
     * @param image The BufferedImage onto which the word will be painted.
     */
    @Override
    public void render(final String word, BufferedImage image) {
        Graphics2D g = image.createGraphics();

        RenderingHints hints = new RenderingHints(
                RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);
        hints.add(new RenderingHints(RenderingHints.KEY_RENDERING,
                RenderingHints.VALUE_RENDER_QUALITY));
        g.setRenderingHints(hints);

        FontRenderContext frc = g.getFontRenderContext();
        int xBaseline = (int) Math.round(image.getWidth() * XOFFSET);
        int yBaseline = image.getHeight() - (int) Math.round(image.getHeight() * YOFFSET);

        char[] chars = new char[1];
        int i = 0;
        for (char c : word.toCharArray()) {
            chars[0] = c;

            g.setColor(_colors.get(RAND.nextInt(_colors.size())));

            int choiceFont = RAND.nextInt(_fonts.size());
            Font font = _fonts.get(choiceFont);
            g.setFont(font);

            GlyphVector gv = font.createGlyphVector(frc, chars);
            if (new Random().nextBoolean()) {
                double theta = (double) i / (double) (word.length() - 1) * Math.PI / 4;
                AffineTransform at = AffineTransform.getTranslateInstance(xBaseline + RAND.nextInt(2), yBaseline + RAND.nextInt(10));
                at.rotate(theta);
                Shape glyph = gv.getOutline();
                Shape transformedGlyph = at.createTransformedShape(glyph);
                g.draw(transformedGlyph);
            } else {
                g.drawGlyphVector(gv, xBaseline + RAND.nextInt(2), yBaseline + RAND.nextInt(10));
            }

            int width = (int) gv.getVisualBounds().getWidth();
            xBaseline = xBaseline + width - 2;
            yBaseline -= +RAND.nextInt(10);
            i++;
        }
    }
}

